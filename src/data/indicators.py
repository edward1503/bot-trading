"""Shared technical-indicator helpers (RSI, MACD, BB, EMA, ATR)."""

import pandas as pd
import ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all indicators used by env/agents to an OHLCV DataFrame."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_pct"] = bb.bollinger_pband()

    df["ema50"] = close.ewm(span=50, min_periods=1, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, min_periods=1, adjust=False).mean()
    df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    df["close_pct"] = close.pct_change().fillna(0.0)
    df["hl_ratio"] = (high - low) / close
    df["bb_pct"] = df["bb_pct"].fillna(0.5)

    df = df.ffill().bfill()
    return df.dropna()
