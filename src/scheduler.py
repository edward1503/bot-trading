"""
24/7 Trading Scheduler.
- Every 5 min: fetch data → LLM agents → RL router → execute
- Every day 00:00: log portfolio snapshot, check circuit breaker
- Every Monday 02:00: run evolutionary RL cycle
"""

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.events import EVENT_JOB_ERROR
from dotenv import load_dotenv

load_dotenv("config/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/bot.log"),
    ],
)
logger = logging.getLogger("scheduler")

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

INSTRUMENT = CONFIG["trading"]["instrument"]
UNITS = CONFIG["trading"]["units"]


# ── Core Trading Loop ─────────────────────────────────────────────────────────

def trading_loop():
    """Main 5-minute trading loop."""
    try:
        from src.data.oanda_fetcher import fetch_candles, fetch_current_price
        from src.data.news_fetcher import fetch_headlines
        from src.agents import technical, sentiment, risk as risk_agent
        from src.execution.oanda_broker import OandaBroker
        from src.router import decide
        from src.db import log_trade, log_portfolio_snapshot

        broker = OandaBroker()
        account = broker.get_account_summary()
        current_pos = broker.get_open_position(INSTRUMENT)

        # Check hard circuit breaker before doing anything
        balance = account.get("balance", 100000)
        daily_pnl_pct = account.get("unrealized_pnl", 0) / balance
        max_dd = CONFIG["risk"]["max_daily_drawdown"]
        if abs(daily_pnl_pct) >= max_dd:
            logger.warning("CIRCUIT BREAKER: daily PnL %+.2f%% exceeds -%d%% limit. Skipping loop.",
                           daily_pnl_pct * 100, max_dd * 100)
            return

        # Fetch data
        df_m5 = fetch_candles(INSTRUMENT, "M5", 200)
        if df_m5.empty:
            logger.warning("Empty M5 data, skipping loop")
            return

        headlines = fetch_headlines(CONFIG["news"]["max_headlines"])

        # LLM agents (3 Groq calls)
        tech = technical.analyze(df_m5, "M5")
        sent = sentiment.analyze(headlines)
        risk = risk_agent.check(
            account, current_pos,
            proposed_signal=tech.get("signal", "hold"),
            confidence=tech.get("confidence", 0.0),
            max_daily_drawdown=max_dd,
        )

        logger.info("LLM: %s@%.2f | Sentiment: bull=%.2f | Risk: veto=%s",
                    tech["signal"], tech["confidence"],
                    sent["bullish_score"], risk["veto"])

        # Signal router → decision
        decision = decide(df_m5, tech, sent, risk, current_pos, account, CONFIG)
        target_units = decision["target_units"]
        current_units = current_pos.get("units", 0)

        logger.info("Decision: %s → target=%d units (current=%d)",
                    decision["action"], target_units, current_units)

        # Execute if position change needed
        trade_result = None
        if decision["action"] != "hold" and target_units != current_units:
            trade_result = broker.adjust_position(INSTRUMENT, target_units, current_units)

        # Log to DB
        price_info = fetch_current_price(INSTRUMENT)
        log_trade({
            "instrument": INSTRUMENT,
            "action": decision["action"],
            "units": target_units,
            "price": price_info["mid"],
            "llm_signal": tech["signal"],
            "llm_reasoning": tech.get("reasoning", ""),
            "llm_confidence": tech["confidence"],
            "rl_action": decision.get("rl_action", 0.0),
            "portfolio_value": account["nav"],
        })

    except Exception as exc:
        logger.exception("Error in trading loop: %s", exc)


# ── Daily Report ──────────────────────────────────────────────────────────────

def daily_report():
    """Log portfolio snapshot at midnight."""
    try:
        from src.execution.oanda_broker import OandaBroker
        from src.db import log_portfolio_snapshot

        broker = OandaBroker()
        account = broker.get_account_summary()
        daily_pnl = account.get("unrealized_pnl", 0.0)

        log_portfolio_snapshot(account, daily_pnl=daily_pnl)
        logger.info("Daily snapshot: balance=%.2f, NAV=%.2f, PnL=%+.2f",
                    account["balance"], account["nav"], daily_pnl)
    except Exception as exc:
        logger.exception("Error in daily report: %s", exc)


# ── Weekly Evolution ──────────────────────────────────────────────────────────

def weekly_evolution():
    """Run evolutionary RL cycle using last 7 days of paper trade data."""
    try:
        logger.info("Starting weekly evolutionary RL cycle...")
        from src.rl.evolution import EvolutionManager
        from src.rl.train import load_historical_data
        from src.db import log_fitness
        from datetime import date, timedelta

        end = date.today().strftime("%Y-%m-%d")
        start = (date.today() - timedelta(days=14)).strftime("%Y-%m-%d")  # 2 weeks for stability

        eval_df = load_historical_data("GC=F", start, end)
        if eval_df.empty:
            logger.warning("No historical data for evolution eval, skipping")
            return

        manager = EvolutionManager(
            base_model_path="models/baseline_ppo",
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

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_listener(on_job_error, EVENT_JOB_ERROR)

    scheduler.add_job(trading_loop, "interval", minutes=interval, id="trading_loop",
                      max_instances=1, coalesce=True)
    scheduler.add_job(daily_report, "cron", hour=0, minute=0, id="daily_report")
    scheduler.add_job(weekly_evolution, "cron", day_of_week="mon", hour=2, minute=0,
                      id="weekly_evolution")

    # Run trading loop immediately on startup
    trading_loop()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user")


if __name__ == "__main__":
    main()
