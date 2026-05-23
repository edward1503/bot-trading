"""Sentiment analyst agent: analyzes gold news headlines via Groq, returns bullish score."""

import os
import json
import time
import logging
from typing import Optional

from groq import Groq

from src.config import load_env

load_env()
logger = logging.getLogger(__name__)

_client: Optional[Groq] = None

# Sentiment of news changes slowly — cache for 1h to cut Groq calls 12x.
_CACHE: dict = {"timestamp": 0.0, "headlines_key": "", "result": None}
_CACHE_TTL_SECONDS = 60 * 60

SYSTEM_PROMPT = """You are a financial sentiment analyst specializing in gold (XAU/USD) markets.
Analyze the provided news headlines and assess overall market sentiment.
Output valid JSON only — no markdown, no explanation outside the JSON."""

SENTIMENT_TEMPLATE = """Recent gold/XAU market news headlines:
{headlines}

Assess overall sentiment for gold price direction.
Return JSON with exactly:
{{"bullish_score": 0.0-1.0, "bearish_score": 0.0-1.0, "summary": "<max 20 words>", "key_theme": "<main driver>"}}

bullish_score + bearish_score should sum to approximately 1.0."""


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def analyze(headlines: list[str], max_retries: int = 3) -> dict:
    """
    Analyze news headlines for gold sentiment.
    Returns: {"bullish_score": float, "bearish_score": float, "summary": str, "key_theme": str}
    Result cached 1h keyed on the headline list.
    """
    if not headlines:
        return _fallback("no headlines")

    headlines_key = "|".join(headlines[:10])
    now = time.time()
    if (_CACHE["result"] is not None
            and now - _CACHE["timestamp"] < _CACHE_TTL_SECONDS
            and _CACHE["headlines_key"] == headlines_key):
        return _CACHE["result"]

    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines[:10]))
    prompt = SENTIMENT_TEMPLATE.format(headlines=numbered)

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
            result = _validate(json.loads(resp.choices[0].message.content))
            _CACHE.update(timestamp=now, headlines_key=headlines_key, result=result)
            return result
        except json.JSONDecodeError as exc:
            logger.warning("Sentiment agent JSON parse failed (attempt %d): %s", attempt + 1, exc)
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as exc:
            logger.error("Sentiment agent error (attempt %d): %s", attempt + 1, exc)
            if attempt < max_retries - 1:
                time.sleep(2)

    return _fallback("max retries exceeded")


def _validate(result: dict) -> dict:
    bullish = float(result.get("bullish_score", 0.5))
    bearish = float(result.get("bearish_score", 0.5))
    bullish = max(0.0, min(1.0, bullish))
    bearish = max(0.0, min(1.0, bearish))
    return {
        "bullish_score": bullish,
        "bearish_score": bearish,
        "net_sentiment": bullish - bearish,   # -1 to 1
        "summary": str(result.get("summary", ""))[:100],
        "key_theme": str(result.get("key_theme", ""))[:60],
    }


def _fallback(reason: str) -> dict:
    logger.warning("Sentiment agent fallback: %s", reason)
    return {"bullish_score": 0.5, "bearish_score": 0.5, "net_sentiment": 0.0,
            "summary": reason, "key_theme": "unavailable"}
