"""Technical analyst agent: sends OHLCV + indicators to Groq, returns structured signal."""

import os
import json
import time
import logging
from typing import Optional

import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv("config/.env")
logger = logging.getLogger(__name__)

_client: Optional[Groq] = None

SYSTEM_PROMPT = """You are an expert gold (XAU/USD) technical analyst at a professional trading firm.
Analyze the provided price action and technical indicators, then output a JSON trading signal.
Be concise and decisive. Output valid JSON only — no markdown, no explanation outside the JSON."""

ANALYSIS_TEMPLATE = """Current XAU/USD Market Data:
- Price: {close:.2f} (open: {open:.2f}, high: {high:.2f}, low: {low:.2f})
- RSI(14): {rsi:.1f}
- MACD: {macd:.4f}, Signal: {macd_signal:.4f}, Histogram: {macd_diff:.4f}
- Bollinger Band position: {bb_pct:.2f} (0=lower band, 1=upper band)
- EMA50: {ema50:.2f}, EMA200: {ema200:.2f}
- ATR(14): {atr:.2f}
- Recent % change: {close_pct:.4f}
- Trend: price is {trend} EMA200

Timeframe: {timeframe}

Return JSON with exactly these fields:
{{"signal": "buy" | "sell" | "hold", "confidence": 0.0-1.0, "size": 0.0-1.0, "reasoning": "<max 25 words>"}}

size guide: 0.25=small, 0.5=medium, 0.75=large, 1.0=full. Scale with conviction strength.
Use size=0.0 when signal=hold."""


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def analyze(df: pd.DataFrame, timeframe: str = "M5", max_retries: int = 3) -> dict:
    """
    Analyze latest candle with indicators.
    Returns: {"signal": "buy"|"sell"|"hold", "confidence": float, "reasoning": str}
    """
    if df.empty:
        return _fallback("empty dataframe")

    row = df.iloc[-1]
    trend = "ABOVE" if row["close"] > row["ema200"] else "BELOW"
    prompt = ANALYSIS_TEMPLATE.format(
        close=row["close"], open=row["open"], high=row["high"], low=row["low"],
        rsi=row["rsi"], macd=row["macd"], macd_signal=row["macd_signal"],
        macd_diff=row["macd_diff"], bb_pct=row["bb_pct"],
        ema50=row["ema50"], ema200=row["ema200"], atr=row["atr"],
        close_pct=row["close_pct"], timeframe=timeframe, trend=trend,
    )

    for attempt in range(max_retries):
        try:
            resp = get_client().chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=120,
                temperature=0.1,
            )
            result = json.loads(resp.choices[0].message.content)
            result = _validate_signal(result)
            logger.debug("Technical signal: %s", result)
            return result
        except json.JSONDecodeError as exc:
            logger.warning("Technical agent JSON parse failed (attempt %d): %s", attempt + 1, exc)
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as exc:
            logger.error("Technical agent error (attempt %d): %s", attempt + 1, exc)
            if attempt < max_retries - 1:
                time.sleep(2)

    return _fallback("max retries exceeded")


def _validate_signal(result: dict) -> dict:
    signal = result.get("signal", "hold").lower()
    if signal not in ("buy", "sell", "hold"):
        signal = "hold"
    confidence = float(result.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    size = float(result.get("size", confidence))   # fallback: dùng confidence làm size
    size = max(0.0, min(1.0, size))
    if signal == "hold":
        size = 0.0
    return {
        "signal":         signal,
        "confidence":     confidence,
        "size":           size,
        "reasoning":      str(result.get("reasoning", ""))[:100],
        "signal_numeric": 1.0 if signal == "buy" else (-1.0 if signal == "sell" else 0.0),
    }


def _fallback(reason: str) -> dict:
    logger.warning("Technical agent fallback: %s", reason)
    return {"signal": "hold", "confidence": 0.0, "reasoning": reason, "signal_numeric": 0.0}
