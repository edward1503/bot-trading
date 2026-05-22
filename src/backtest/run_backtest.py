"""
Backtesting wrapper using backtesting.py library.
Wraps the trained PPO model into a Strategy that backtesting.py can evaluate.

Usage:
  python src/backtest/run_backtest.py \
    --start 2024-01-01 --end 2024-06-01 \
    --model models/baseline_ppo
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from stable_baselines3 import PPO

from src.rl.train import load_historical_data
from src.rl.env import XAUUSDTradingEnv
from src.data.oanda_fetcher import _add_indicators

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class PPOStrategy(Strategy):
    model_path = "models/baseline_ppo"

    def init(self):
        self._model = PPO.load(self.model_path)
        self._env_df = None  # set in run_backtest before Backtest.run()

    def next(self):
        if self._env_df is None:
            return

        idx = len(self.data) - 1
        if idx >= len(self._env_df):
            return

        # Build a mini-env just for observation
        env = XAUUSDTradingEnv(self._env_df.iloc[max(0, idx - 200): idx + 1])
        obs, _ = env.reset()
        action, _ = self._model.predict(obs, deterministic=True)
        position_target = float(action[0])  # -1 to 1

        current_position = self.position.size / (self.equity / self.data.Close[-1] + 1e-8)

        if position_target > 0.1 and not self.position.is_long:
            self.buy(size=min(0.95, position_target))
        elif position_target < -0.1 and not self.position.is_short:
            self.sell(size=min(0.95, abs(position_target)))
        elif abs(position_target) <= 0.1 and self.position:
            self.position.close()


def run_backtest(
    start: str,
    end: str,
    model_path: str = "models/baseline_ppo",
    symbol: str = "GC=F",
    cash: float = 100_000.0,
    commission: float = 0.0002,
) -> dict:
    df_raw = load_historical_data(symbol, start, end)

    # backtesting.py needs capitalized OHLCV columns
    bt_df = df_raw[["open", "high", "low", "close", "volume"]].copy()
    bt_df.columns = ["Open", "High", "Low", "Close", "Volume"]
    bt_df.index = pd.DatetimeIndex(bt_df.index)

    PPOStrategy.model_path = model_path
    PPOStrategy._env_df_ref = df_raw  # passed via class attribute

    # Monkey-patch init to attach env_df after model load
    original_init = PPOStrategy.init
    def patched_init(self):
        original_init(self)
        self._env_df = df_raw
    PPOStrategy.init = patched_init

    bt = Backtest(bt_df, PPOStrategy, cash=cash, commission=commission, exclusive_orders=True)
    stats = bt.run()

    logger.info("\n%s", stats)

    # Save HTML report
    os.makedirs("results", exist_ok=True)
    report_path = f"results/backtest_{start}_{end}.html"
    bt.plot(filename=report_path, open_browser=False)
    logger.info("Report saved to %s", report_path)

    metrics = {
        "start": start,
        "end": end,
        "return_pct": float(stats["Return [%]"]),
        "sharpe": float(stats["Sharpe Ratio"]),
        "max_drawdown_pct": float(stats["Max. Drawdown [%]"]),
        "win_rate_pct": float(stats["Win Rate [%]"]),
        "trades": int(stats["# Trades"]),
        "report": report_path,
    }
    return metrics


def walk_forward(
    windows: int = 4,
    train_months: int = 18,
    test_months: int = 6,
    symbol: str = "GC=F",
):
    """Walk-forward validation: rolling train/test windows."""
    from dateutil.relativedelta import relativedelta
    from datetime import date

    start_date = date(2022, 1, 1)
    results = []

    for i in range(windows):
        train_start = (start_date + relativedelta(months=i * test_months)).strftime("%Y-%m-%d")
        train_end = (start_date + relativedelta(months=i * test_months + train_months)).strftime("%Y-%m-%d")
        test_start = train_end
        test_end = (start_date + relativedelta(months=i * test_months + train_months + test_months)).strftime("%Y-%m-%d")

        model_path = f"models/wf_ppo_window{i}"
        logger.info("Window %d: train %s→%s, test %s→%s", i, train_start, train_end, test_start, test_end)

        from src.rl.train import train
        train(train_start, train_end, timesteps=500_000, output=model_path, symbol=symbol)

        metrics = run_backtest(test_start, test_end, model_path=model_path, symbol=symbol)
        metrics["window"] = i
        results.append(metrics)
        logger.info("Window %d Sharpe: %.3f", i, metrics["sharpe"])

    avg_sharpe = sum(r["sharpe"] for r in results) / len(results)
    logger.info("\n=== Walk-Forward Results ===")
    for r in results:
        logger.info("  Window %d: Sharpe=%.3f, Return=%.1f%%, MaxDD=%.1f%%",
                    r["window"], r["sharpe"], r["return_pct"], r["max_drawdown_pct"])
    logger.info("Average Sharpe: %.3f", avg_sharpe)
    return results, avg_sharpe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-06-01")
    parser.add_argument("--model", default="models/baseline_ppo")
    parser.add_argument("--symbol", default="GC=F")
    parser.add_argument("--walk-forward", action="store_true")
    args = parser.parse_args()

    if args.walk_forward:
        walk_forward()
    else:
        metrics = run_backtest(args.start, args.end, model_path=args.model, symbol=args.symbol)
        print("\nBacktest Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
