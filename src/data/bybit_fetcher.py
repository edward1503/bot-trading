"""Fetch OHLCV candles and current price for XAUUSDT.

Market data (klines/price) comes from Bybit MAINNET public API — no auth needed,
real prices. Execution (orders/account) uses testnet via BybitBroker.
"""

import os
import logging
from typing import Optional

import pandas as pd
import ta
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv("config/.env")
logger = logging.getLogger(__name__)

# Mainnet session for public market data (real prices, no auth required)
_market_session: Optional[HTTP] = None
# Testnet session kept for compatibility if needed
_session: Optional[HTTP] = None


def get_market_session() -> HTTP:
    """Mainnet public session — real XAUUSDT price data, no credentials needed."""
    global _market_session
    if _market_session is None:
        _market_session = HTTP(testnet=False)
    return _market_session


def get_session() -> HTTP:
    """Testnet session (legacy, used by execution layer)."""
    global _session
    if _session is None:
        _session = HTTP(
            testnet=True,
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
        )
    return _session


def fetch_candles(
    symbol: str = "XAUUSDT",
    interval: str = "5",      # "1","3","5","15","30","60","120","240","D"
    limit: int = 1000,         # max Bybit allows; need >=200 for EMA200
) -> pd.DataFrame:
    """Return OHLCV DataFrame with technical indicators from Bybit mainnet."""
    resp = get_market_session().get_kline(
        category="linear",
        symbol=symbol,
        interval=interval,
        limit=min(limit, 1000),
    )
    if resp["retCode"] != 0:
        raise RuntimeError(f"Bybit kline error: {resp['retMsg']}")

    rows = []
    for bar in reversed(resp["result"]["list"]):   # reverse: oldest first
        rows.append({
            "time": pd.Timestamp(int(bar[0]), unit="ms", tz="UTC"),
            "open":   float(bar[1]),
            "high":   float(bar[2]),
            "low":    float(bar[3]),
            "close":  float(bar[4]),
            "volume": float(bar[5]),
        })

    df = pd.DataFrame(rows).set_index("time")
    if df.empty:
        logger.warning("fetch_candles returned empty DataFrame for %s %s", symbol, interval)
        return df

    return _add_indicators(df)


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    df["rsi"]         = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd              = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()

    bb                = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_mid"]      = bb.bollinger_mavg()
    df["bb_pct"]      = bb.bollinger_pband()   # 0=lower band, 1=upper band

    # min_periods=1 so indicators return values even when fewer bars than window
    df["ema50"]       = close.ewm(span=50,  min_periods=1, adjust=False).mean()
    df["ema200"]      = close.ewm(span=200, min_periods=1, adjust=False).mean()
    df["atr"]         = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    df["close_pct"]   = close.pct_change().fillna(0.0)
    df["hl_ratio"]    = (high - low) / close

    # bb_pct is NaN when std=0 (constant price) — fill with 0.5 (neutral)
    df["bb_pct"]      = df["bb_pct"].fillna(0.5)

    # Forward-fill then back-fill remaining NaNs from short-window warmup
    df = df.ffill().bfill()
    return df.dropna()


def fetch_current_price(symbol: str = "XAUUSDT") -> dict:
    """Return latest mark/last price from mainnet (real price)."""
    resp = get_market_session().get_tickers(category="linear", symbol=symbol)
    if resp["retCode"] != 0:
        raise RuntimeError(f"Bybit ticker error: {resp['retMsg']}")
    ticker = resp["result"]["list"][0]
    bid = float(ticker.get("bid1Price") or ticker["lastPrice"])
    ask = float(ticker.get("ask1Price") or ticker["lastPrice"])
    last = float(ticker["lastPrice"])
    return {
        "symbol":  symbol,
        "bid":     bid,
        "ask":     ask,
        "mid":     round((bid + ask) / 2, 2),
        "last":    last,
        "spread":  round(ask - bid, 2),
        "mark":    float(ticker.get("markPrice", last)),
    }


def fetch_multi_timeframe(
    symbol: str = "XAUUSDT",
    timeframes: list[str] = None,
    limit: int = 200,
) -> dict[str, pd.DataFrame]:
    if timeframes is None:
        timeframes = ["5", "15", "60"]
    return {tf: fetch_candles(symbol, tf, limit) for tf in timeframes}
