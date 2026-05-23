"""
24/7 Trading Scheduler.
- Every 5 min: fetch data → LLM agents → RL router → execute
- Every day 00:00: log portfolio snapshot, check circuit breaker
- Every Monday 02:00: run evolutionary RL cycle
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.events import EVENT_JOB_ERROR

from src.config import PROJECT_ROOT, load_config, load_env

load_env()

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "bot.log"),
    ],
)
logger = logging.getLogger("scheduler")

CONFIG = load_config()

INSTRUMENT = CONFIG["trading"]["instrument"]
QTY = CONFIG["trading"]["qty"]


# ── Core Trading Loop ─────────────────────────────────────────────────────────

def trading_loop():
    """Main 5-minute trading loop."""
    try:
        from src.data.bybit_fetcher import fetch_candles, fetch_current_price
        from src.data.news_fetcher import get_cached_headlines
        from src.agents import technical, sentiment, risk as risk_agent
        from src.execution.paper_trader import PaperTrader
        from src.router import decide
        from src.db import log_trade, log_portfolio_snapshot, get_daily_drawdown

        broker = PaperTrader(symbol=INSTRUMENT)
        account = broker.get_account_summary()
        current_pos = broker.get_open_position(INSTRUMENT)

        # Daily DD circuit breaker: triggers on NAV drop from today's peak.
        max_dd = CONFIG["risk"]["max_daily_drawdown"]
        dd_info = get_daily_drawdown(account["nav"])
        if dd_info["drawdown_pct"] >= max_dd:
            logger.warning(
                "CIRCUIT BREAKER: daily DD %.2f%% ≥ %.0f%% (peak=$%.2f, nav=$%.2f). Skipping loop.",
                dd_info["drawdown_pct"] * 100, max_dd * 100, dd_info["peak_nav"], account["nav"],
            )
            return

        # Fetch M5 candles (interval "5" = 5min on Bybit)
        tf_main = CONFIG["trading"]["timeframes"][0]   # "5"
        df_m5 = fetch_candles(INSTRUMENT, tf_main, 200)
        if df_m5.empty:
            logger.warning("Empty candle data, skipping loop")
            return

        headlines = get_cached_headlines(CONFIG["news"]["max_headlines"])

        # LLM agents. Tech every loop; sentiment is internally cached 1h.
        # Risk agent only called when tech proposes an actual trade — saves ~70% of calls.
        tech = technical.analyze(df_m5, f"M{tf_main}")
        sent = sentiment.analyze(headlines)
        proposed_signal = tech.get("signal", "hold")
        if proposed_signal == "hold":
            risk = {"veto": False, "reason": "skipped (hold)", "adjusted_size_pct": 0.0}
        else:
            risk = risk_agent.check(
                account, current_pos,
                proposed_signal=proposed_signal,
                confidence=tech.get("confidence", 0.0),
                max_daily_drawdown=max_dd,
            )

        logger.info("LLM: %s@%.2f size=%.2f | Sentiment: bull=%.2f | Risk: veto=%s (%s)",
                    tech["signal"], tech["confidence"], tech.get("size", 0),
                    sent["bullish_score"], risk["veto"], risk.get("reason", ""))

        # Signal router → decision
        decision = decide(df_m5, tech, sent, risk, current_pos, account, CONFIG)
        target_size = decision["target_units"]    # -1 to 1 float from router
        current_size = current_pos.get("size", 0.0)

        logger.info("Decision: %s → target_size=%.3f (current=%.3f)",
                    decision["action"], target_size, current_size)

        # Fetch mainnet price (real market price, dùng cho PnL thật)
        price_info = fetch_current_price(INSTRUMENT)
        mainnet_price = price_info["last"]

        # Execute — qty scaled by LLM size
        action = decision["action"]
        actual_oz = 0.0
        if action in ("buy", "sell"):
            llm_size = tech.get("size", 0.5)
            scaled_qty = round(max(QTY, QTY * 10 * llm_size), 3)  # min=QTY, max=QTY×10
            actual_oz = round(scaled_qty * 10, 3)
            broker.adjust_position(
                INSTRUMENT, target_size, current_size, base_qty=scaled_qty
            )
            logger.info("Executing %s: %.3f oz (LLM size=%.2f, notional=$%.0f)",
                        action, actual_oz, llm_size, actual_oz * mainnet_price)
        elif action == "close":
            broker.close_position(INSTRUMENT)

        # Re-read state after execution — PaperTrader persists to paper_portfolio table
        post_account = broker.get_account_summary()
        real_pnl = post_account["unrealized_pnl"]
        real_nav = post_account["nav"]

        logger.info("PnL: $%+.2f | NAV: $%.2f (balance=$%.2f, realized=$%+.2f)",
                    real_pnl, real_nav, post_account["balance"], post_account.get("realized_pnl", 0.0))

        # Log mọi loop vào trade log
        log_trade({
            "instrument":    INSTRUMENT,
            "action":        action,
            "units":         int(round(target_size * 100)),
            "volume_oz":     actual_oz if action in ("buy", "sell") else 0.0,
            "price":         mainnet_price,
            "llm_signal":    tech["signal"],
            "llm_reasoning": tech.get("reasoning", ""),
            "llm_confidence": tech["confidence"],
            "rl_action":     decision.get("rl_action", 0.0),
            "portfolio_value": real_nav,
        })

        # Portfolio snapshot với PnL thật mỗi loop
        log_portfolio_snapshot(
            {"balance": post_account["balance"], "nav": real_nav, "unrealized_pnl": real_pnl},
            daily_pnl=real_pnl,
        )

    except Exception as exc:
        logger.exception("Error in trading loop: %s", exc)


# ── Daily Report ──────────────────────────────────────────────────────────────

def daily_report():
    """Log portfolio snapshot at midnight."""
    try:
        from src.execution.paper_trader import PaperTrader
        from src.db import log_portfolio_snapshot

        broker = PaperTrader(symbol=INSTRUMENT)
        account = broker.get_account_summary()
        daily_pnl = account.get("unrealized_pnl", 0.0)

        log_portfolio_snapshot(account, daily_pnl=daily_pnl)
        logger.info("Daily snapshot: balance=%.2f, NAV=%.2f, PnL=%+.2f",
                    account["balance"], account["nav"], daily_pnl)
    except Exception as exc:
        logger.exception("Error in daily report: %s", exc)


# ── Daily Fine-tune ───────────────────────────────────────────────────────────

def daily_finetune():
    """Fine-tune PPO on yesterday's data at 01:00 UTC."""
    try:
        from src.rl.daily_trainer import run_daily_finetune
        result = run_daily_finetune()
        logger.info("Daily fine-tune result: %s", result)
    except Exception as exc:
        logger.exception("Error in daily fine-tune: %s", exc)


# ── Weekly Evolution ──────────────────────────────────────────────────────────

def weekly_evolution():
    """Run evolutionary RL cycle using last 7 days of paper trade data."""
    try:
        logger.info("Starting weekly evolutionary RL cycle...")
        from src.rl.evolution import EvolutionManager
        from src.data.bybit_fetcher import fetch_historical_candles
        from src.db import log_fitness

        # Evolve on the same instrument+timeframe the bot actually trades:
        # 14 days of Bybit XAUUSDT M5 (~4000 bars).
        eval_df = fetch_historical_candles(INSTRUMENT, "5", days=14)
        if eval_df.empty or len(eval_df) < 200:
            logger.warning("Insufficient Bybit history for evolution (%d bars), skipping",
                           len(eval_df))
            return

        manager = EvolutionManager(
            base_model_path=str(PROJECT_ROOT / "models" / "baseline_ppo"),
            pop_size=CONFIG["rl"]["population_size"],
        )
        manager.load_state()
        cycle_result = manager.run_weekly_cycle(eval_df)
        log_fitness(cycle_result)

        logger.info("Evolution complete: gen=%d, best_sharpe=%.3f",
                    cycle_result["generation"], cycle_result["best_sharpe"])

        # Reload policy in router
        from src.router import reload_policy
        reload_policy()

    except Exception as exc:
        logger.exception("Error in weekly evolution: %s", exc)


# ── Error handler ─────────────────────────────────────────────────────────────

def on_job_error(event):
    logger.error("Scheduler job failed: %s — %s", event.job_id, event.exception)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    interval = CONFIG["trading"]["loop_interval_minutes"]
    logger.info("Starting XAUUSD bot | instrument=%s | loop=%dmin", INSTRUMENT, interval)

    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    (PROJECT_ROOT / "models").mkdir(exist_ok=True)

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_listener(on_job_error, EVENT_JOB_ERROR)

    scheduler.add_job(trading_loop,    "interval", minutes=interval, id="trading_loop",
                      max_instances=1, coalesce=True)
    scheduler.add_job(daily_report,    "cron", hour=0,  minute=0,  id="daily_report")
    scheduler.add_job(daily_finetune,  "cron", hour=1,  minute=0,  id="daily_finetune")
    scheduler.add_job(weekly_evolution,"cron", hour=2,  minute=0,
                      day_of_week="mon", id="weekly_evolution")

    # Run trading loop immediately on startup
    trading_loop()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user")


if __name__ == "__main__":
    main()
