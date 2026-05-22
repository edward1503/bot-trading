"""Fetch OHLCV candles and streaming prices from OANDA v20 API."""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import ta
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.pricing as pricing
from dotenv import load_dotenv

load_dotenv("config/.env")
logger = logging.getLogger(__name__)

_client: Optional[oandapyV20.API] = None


def get_client() -> oandapyV20.API:
    global _client
    if _client is None:
        _client = oandapyV20.API(
            access_token=os.getenv("OANDA_API_KEY"),
            environment="practice",
        )
    return _client


def fetch_candles(
    instrument: str = "XAU_USD",
    granularity: str = "M5",
    count: int = 200,
) -> pd.DataFrame:
    """Return OHLCV DataFrame with technical indicators attached."""
    params = {"granularity": granularity, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument, params=params)
    get_client().request(r)

    rows = []
    for candle in r.response["candles"]:
        if not candle["complete"]:
            continue
        mid = candle["mid"]
        rows.append({
            "time": pd.Timestamp(candle["time"]),
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low": float(mid["l"]),
            "close": float(mid["c"]),
            "volume": int(candle["volume"]),
        })

    df = pd.DataFrame(rows).set_index("time")
    if df.empty:
        logger.warning("fetch_candles returned empty DataFrame for %s %s", instrument, granularity)
        return df

    return _add_indicators(df)


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
    df["bb_pct"] = bb.bollinger_pband()   # 0=at lower, 1=at upper

    df["ema50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(close, window=200).ema_indicator()
    df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    df["close_pct"] = close.pct_change()
    df["hl_ratio"] = (high - low) / close

    return df.dropna()


def fetch_current_price(instrument: str = "XAU_USD") -> dict:
    """Return latest bid/ask/mid price for the instrument."""
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    params = {"instruments": instrument}
    r = pricing.PricingInfo(account_id, params=params)
    get_client().request(r)
    price_data = r.response["prices"][0]
    bid = float(price_data["bids"][0]["price"])
    ask = float(price_data["asks"][0]["price"])
    return {
        "instrument": instrument,
        "bid": bid,
        "ask": ask,
        "mid": round((bid + ask) / 2, 5),
        "spread": round(ask - bid, 5),
        "time": price_data["time"],
    }


def fetch_multi_timeframe(
    instrument: str = "XAU_USD",
    timeframes: list[str] = None,
    count: int = 200,
) -> dict[str, pd.DataFrame]:
    """Fetch candles for multiple timeframes, return dict keyed by granularity."""
    if timeframes is None:
        timeframes = ["M5", "M15", "H1"]
    return {tf: fetch_candles(instrument, tf, count) for tf in timeframes}
