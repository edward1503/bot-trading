"""
XAUUSD Gym trading environment for PPO training.
Observation: 20-dim vector (OHLCV-derived + indicators + LLM signals + portfolio state)
Action: continuous position in [-1, 1] (short → flat → long)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class XAUUSDTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    TRANSACTION_COST = 0.0002   # 2 pips per trade (spread)
    DRAWDOWN_PENALTY = 0.1      # penalty coefficient when drawdown > 2%

    def __init__(self, df: pd.DataFrame, initial_cash: float = 100_000.0):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.n_steps = len(df)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
        # Continuous position: -1.0 = full short, 0.0 = flat, 1.0 = full long
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._reset_state()

    def _reset_state(self):
        self.step_idx = 0
        self.position = 0.0          # current position (-1 to 1)
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.peak_value = self.initial_cash
        self.entry_price = 0.0
        self.bars_in_trade = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        new_position = float(np.clip(action[0], -1.0, 1.0))
        row = self.df.iloc[self.step_idx]
        price = float(row["close"])

        # Transaction cost proportional to position change
        position_delta = abs(new_position - self.position)
        transaction_cost = position_delta * self.TRANSACTION_COST * self.initial_cash

        # PnL from holding current position
        if self.step_idx > 0:
            prev_price = float(self.df.iloc[self.step_idx - 1]["close"])
            price_return = (price - prev_price) / prev_price
            pnl = self.position * price_return * self.portfolio_value
        else:
            pnl = 0.0

        self.position = new_position
        self.portfolio_value = self.portfolio_value + pnl - transaction_cost
        self.portfolio_value = max(self.portfolio_value, 1.0)  # avoid zero

        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        # Reward: portfolio return - transaction cost - drawdown penalty
        reward = (pnl - transaction_cost) / self.initial_cash
        if drawdown > 0.02:
            reward -= self.DRAWDOWN_PENALTY * drawdown

        self.step_idx += 1
        self.bars_in_trade = self.bars_in_trade + 1 if self.position != 0 else 0

        terminated = self.step_idx >= self.n_steps - 1
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(20, dtype=np.float32)
        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "drawdown": drawdown,
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.step_idx]
        price = float(row["close"])

        # Normalize relative to current price
        ema50_pct = (float(row["ema50"]) - price) / price if price > 0 else 0.0
        ema200_pct = (float(row["ema200"]) - price) / price if price > 0 else 0.0
        atr_norm = float(row["atr"]) / price if price > 0 else 0.0

        obs = np.array([
            # Price action
            float(row.get("close_pct", 0.0)),           # 0
            float(row.get("hl_ratio", 0.0)),             # 1
            # Indicators (normalized to 0-1 or -1 to 1)
            float(row.get("rsi", 50.0)) / 100.0,         # 2
            float(row.get("macd", 0.0)) / (price * 0.01 + 1e-8),  # 3
            float(row.get("macd_diff", 0.0)) / (price * 0.01 + 1e-8),  # 4
            float(row.get("bb_pct", 0.5)),               # 5
            ema50_pct,                                   # 6
            ema200_pct,                                  # 7
            atr_norm,                                    # 8
            # Session (approximate from index position — 0-1 cyclic)
            float(self.step_idx % 288) / 288.0,          # 9 intra-day position
            # LLM signals (placeholder: filled by router during live trading)
            0.0,                                         # 10 llm_tech_signal_numeric
            0.5,                                         # 11 llm_tech_confidence
            0.5,                                         # 12 llm_sentiment_score
            # Portfolio state
            self.position,                               # 13 current position
            (self.portfolio_value - self.initial_cash) / self.initial_cash,  # 14 total return
            (self.peak_value - self.portfolio_value) / self.peak_value,       # 15 drawdown
            float(min(self.bars_in_trade, 100)) / 100.0, # 16 time in trade
            # Padding
            0.0, 0.0, 0.0,
        ], dtype=np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
