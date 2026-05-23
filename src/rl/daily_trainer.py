"""
Daily fine-tuning: fine-tune PPO on yesterday's XAUUSD data.
Called every day at 01:00 UTC by scheduler.

Flow:
  1. Download yesterday's 1h candles (GC=F via yfinance)
  2. Append to cumulative training buffer (CSV)
  3. Fine-tune existing model 50k more steps on recent data
  4. Only deploy if new model Sharpe >= current model Sharpe
"""

import logging
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

logger = logging.getLogger(__name__)

BUFFER_PATH    = "data/historical/cumulative_buffer.csv"
BEST_PATH      = "models/best_policy"
BASELINE_PATH  = "models/baseline_ppo"
FINETUNE_STEPS = 50_000


def _active_model_path() -> str:
    if os.path.exists(BEST_PATH + ".zip"):
        return BEST_PATH
    if os.path.exists(BASELINE_PATH + ".zip"):
        return BASELINE_PATH
    return None


def _fetch_yesterday(symbol: str = "GC=F") -> pd.DataFrame:
    """Download yesterday's 1h OHLCV from yfinance."""
    import yfinance as yf
    from src.data.indicators import add_indicators

    yesterday = date.today() - timedelta(days=1)
    # yfinance: fetch a few days back to ensure we get data
    start = (yesterday - timedelta(days=3)).isoformat()
    end   = date.today().isoformat()

    df = yf.Ticker(symbol).history(start=start, end=end, interval="1h")
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return add_indicators(df)


def _append_to_buffer(new_df: pd.DataFrame) -> pd.DataFrame:
    """Append new rows to cumulative CSV buffer, return full buffer."""
    os.makedirs(os.path.dirname(BUFFER_PATH), exist_ok=True)
    if os.path.exists(BUFFER_PATH):
        existing = pd.read_csv(BUFFER_PATH, index_col=0, parse_dates=True)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = new_df
    combined.to_csv(BUFFER_PATH)
    return combined


def _sharpe(model: PPO, df: pd.DataFrame) -> float:
    """Quick Sharpe estimate: run model on df, return annualised Sharpe."""
    from src.rl.env import XAUUSDTradingEnv
    env  = XAUUSDTradingEnv(df)
    obs, _ = env.reset()
    vals = [env.initial_cash]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        vals.append(info["portfolio_value"])
        done = terminated or truncated
    rets = np.diff(vals) / np.array(vals[:-1])
    if rets.std() < 1e-8:
        return 0.0
    return float((rets.mean() / rets.std()) * np.sqrt(6000))


def run_daily_finetune(symbol: str = "GC=F") -> dict:
    """
    Main entry point called by scheduler at 01:00 UTC daily.
    Returns dict with result summary.
    """
    logger.info("Daily fine-tune started (%s)", date.today().isoformat())

    model_path = _active_model_path()
    if model_path is None:
        logger.warning("No trained model found — skipping fine-tune. Run train.py first.")
        return {"status": "skipped", "reason": "no model"}

    # 1. Fetch new data
    try:
        new_df = _fetch_yesterday(symbol)
        if new_df.empty or len(new_df) < 10:
            logger.warning("Not enough new data (%d rows) — skipping", len(new_df))
            return {"status": "skipped", "reason": "insufficient data"}
        logger.info("Fetched %d new bars", len(new_df))
    except Exception as exc:
        logger.error("Data fetch failed: %s", exc)
        return {"status": "error", "reason": str(exc)}

    # 2. Append to buffer
    full_df = _append_to_buffer(new_df)
    # Use rolling window: last 90 days to avoid overfitting on old regimes
    recent_df = full_df.iloc[-90 * 24:]   # ~90 days of 1h bars
    logger.info("Training buffer: %d bars (recent %d)", len(full_df), len(recent_df))

    # 3. Load current model and evaluate baseline Sharpe
    current_model = PPO.load(model_path)
    sharpe_before = _sharpe(current_model, recent_df)
    logger.info("Sharpe before fine-tune: %.3f", sharpe_before)

    # 4. Fine-tune
    from src.rl.env import XAUUSDTradingEnv
    env = make_vec_env(lambda: XAUUSDTradingEnv(recent_df), n_envs=1)
    current_model.set_env(env)
    current_model.learn(
        total_timesteps=FINETUNE_STEPS,
        reset_num_timesteps=False,   # continue from existing timestep count
    )

    # 5. Evaluate fine-tuned Sharpe
    sharpe_after = _sharpe(current_model, recent_df)
    logger.info("Sharpe after fine-tune:  %.3f", sharpe_after)

    # 6. Deploy only if improved (or at most 5% worse — allow exploration)
    improved = sharpe_after >= sharpe_before * 0.95
    if improved:
        current_model.save(BEST_PATH)
        logger.info("✓ Fine-tuned model deployed (Sharpe %.3f → %.3f)", sharpe_before, sharpe_after)
        # Reload router policy
        from src.router import reload_policy
        reload_policy()
        status = "deployed"
    else:
        logger.info("✗ Fine-tuned model NOT deployed (Sharpe %.3f → %.3f, regression too large)",
                    sharpe_before, sharpe_after)
        status = "rejected"

    return {
        "status":        status,
        "sharpe_before": sharpe_before,
        "sharpe_after":  sharpe_after,
        "bars_used":     len(recent_df),
        "date":          date.today().isoformat(),
    }
