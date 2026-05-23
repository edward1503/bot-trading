"""Train PPO baseline on historical XAU/USD data."""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from src.rl.env import XAUUSDTradingEnv
from src.data.indicators import add_indicators

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_historical_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download historical data via yfinance (GC=F = Gold Futures)."""
    cache_path = f"data/historical/{symbol.replace('=','').replace('/','_')}_{start}_{end}.csv"
    if os.path.exists(cache_path):
        logger.info("Loading cached data from %s", cache_path)
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        logger.info("Downloading %s from %s to %s ...", symbol, start, end)
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval="1h")
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        os.makedirs("data/historical", exist_ok=True)
        df.to_csv(cache_path)
        logger.info("Saved to %s (%d rows)", cache_path, len(df))

    return add_indicators(df)


def train(
    train_start: str,
    train_end: str,
    timesteps: int,
    output: str,
    symbol: str = "GC=F",
) -> PPO:
    df_train = load_historical_data(symbol, train_start, train_end)
    logger.info("Training on %d bars (%s → %s)", len(df_train), train_start, train_end)

    env = make_vec_env(lambda: XAUUSDTradingEnv(df_train), n_envs=1)
    eval_env = make_vec_env(lambda: XAUUSDTradingEnv(df_train), n_envs=1)

    callbacks = [
        EvalCallback(eval_env, best_model_save_path="models/", log_path="logs/",
                     eval_freq=50_000, deterministic=True, render=False),
        CheckpointCallback(save_freq=100_000, save_path="models/", name_prefix="ppo_xauusd"),
    ]

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="logs/tensorboard/",
    )

    logger.info("Training PPO for %d timesteps ...", timesteps)
    model.learn(total_timesteps=timesteps, callback=callbacks)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    model.save(output)
    logger.info("Model saved to %s", output)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on XAUUSD historical data")
    parser.add_argument("--train-start", default="2022-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--output", default="models/baseline_ppo")
    parser.add_argument("--symbol", default="GC=F")
    args = parser.parse_args()

    train(args.train_start, args.train_end, args.timesteps, args.output, args.symbol)
