"""Fetch OHLCV candles and current price for XAUUSDT.

Market data (klines/price) comes from Bybit MAINNET public API — no auth needed,
real prices. Execution (orders/account) uses testnet via BybitBroker.
"""

import os
import logging
from typing import Optional

import pandas as pd
from pybit.unified_trading import HTTP

from src.config import load_env
from src.data.indicators import add_indicators

load_env()
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

    return add_indicators(df)


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


def fetch_historical_candles(
    symbol: str = "XAUUSDT",
    interval: str = "5",
    days: int = 14,
) -> pd.DataFrame:
    """Fetch up to N days of OHLCV via pagination (Bybit limits each call to 1000 bars)."""
    import time as _time

    now_ms = int(_time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000
    interval_minutes = {"1": 1, "3": 3, "5": 5, "15": 15, "30": 30, "60": 60,
                        "120": 120, "240": 240, "D": 1440}.get(interval, 5)
    bar_ms = interval_minutes * 60 * 1000

    session = get_market_session()
    all_rows: list = []
    cursor_end = now_ms
    while cursor_end > start_ms:
        resp = session.get_kline(
            category="linear", symbol=symbol, interval=interval,
            end=cursor_end, limit=1000,
        )
        if resp["retCode"] != 0:
            raise RuntimeError(f"Bybit kline error: {resp['retMsg']}")
        bars = resp["result"]["list"]
        if not bars:
            break
        all_rows.extend(bars)
        oldest_ms = int(bars[-1][0])
        if oldest_ms <= start_ms or oldest_ms == cursor_end:
            break
        cursor_end = oldest_ms - bar_ms

    if not all_rows:
        return pd.DataFrame()

    # Deduplicate by timestamp, then sort ascending
    seen = {}
    for bar in all_rows:
        seen[int(bar[0])] = bar
    sorted_bars = [seen[k] for k in sorted(seen.keys())]

    rows = [{
        "time": pd.Timestamp(int(b[0]), unit="ms", tz="UTC"),
        "open": float(b[1]), "high": float(b[2]), "low": float(b[3]),
        "close": float(b[4]), "volume": float(b[5]),
    } for b in sorted_bars]
    df = pd.DataFrame(rows).set_index("time")
    df = df[df.index >= pd.Timestamp(start_ms, unit="ms", tz="UTC")]
    return add_indicators(df)


def fetch_multi_timeframe(
    symbol: str = "XAUUSDT",
    timeframes: list[str] = None,
    limit: int = 200,
) -> dict[str, pd.DataFrame]:
    if timeframes is None:
        timeframes = ["5", "15", "60"]
    return {tf: fetch_candles(symbol, tf, limit) for tf in timeframes}
