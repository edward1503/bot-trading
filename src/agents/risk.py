"""Risk manager agent: checks portfolio health and vetoes unsafe trades via Groq."""

import os
import json
import time
import logging
from typing import Optional

from groq import Groq
from dotenv import load_dotenv

load_dotenv("config/.env")
logger = logging.getLogger(__name__)

_client: Optional[Groq] = None

SYSTEM_PROMPT = """You are a strict risk manager at a gold trading desk.
Your job is to protect capital. Review portfolio state and proposed trade.
Output valid JSON only — no markdown, no explanation outside the JSON."""

RISK_TEMPLATE = """Portfolio State:
- Balance: ${balance:,.2f}
- Unrealized PnL today: ${daily_pnl:+.2f} ({daily_pnl_pct:+.2f}%)
- Current XAU/USD position: {units} units (positive=long, negative=short, 0=flat)
- Max allowed daily drawdown: {max_dd_pct:.0%}
- Portfolio drawdown from peak: {portfolio_dd_pct:.2f}%

Proposed action: {proposed_signal} with confidence {confidence:.0%}

Assess risk and decide whether to allow this trade.
Return JSON with exactly:
{{"veto": true|false, "reason": "<max 20 words>", "adjusted_size_pct": 0.0-1.0}}

adjusted_size_pct: 1.0 = full size, 0.5 = half size, 0.0 = no trade"""


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def check(
    account_summary: dict,
    current_position: dict,
    proposed_signal: str,
    confidence: float,
    max_daily_drawdown: float = 0.03,
    max_retries: int = 3,
) -> dict:
    """
    Check if proposed trade is safe.
    Returns: {"veto": bool, "reason": str, "adjusted_size_pct": float}
    """
    balance = account_summary.get("balance", 100000)
    nav = account_summary.get("nav", balance)
    daily_pnl = account_summary.get("unrealized_pnl", 0.0)
    daily_pnl_pct = (daily_pnl / balance) * 100 if balance > 0 else 0.0
    portfolio_dd_pct = max(0.0, ((balance - nav) / balance) * 100) if balance > 0 else 0.0

    # Hard circuit breaker: bypass LLM if drawdown already exceeded
    if abs(daily_pnl_pct / 100) >= max_daily_drawdown:
        reason = f"Circuit breaker: daily drawdown {daily_pnl_pct:.1f}% exceeded {max_daily_drawdown:.0%} limit"
        logger.warning("Risk VETO (circuit breaker): %s", reason)
        return {"veto": True, "reason": reason, "adjusted_size_pct": 0.0}

    prompt = RISK_TEMPLATE.format(
        balance=balance,
        daily_pnl=daily_pnl,
        daily_pnl_pct=daily_pnl_pct,
        units=current_position.get("units", 0),
        max_dd_pct=max_daily_drawdown,
        portfolio_dd_pct=portfolio_dd_pct,
        proposed_signal=proposed_signal,
        confidence=confidence,
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
                max_tokens=100,
                temperature=0.05,
            )
            result = json.loads(resp.choices[0].message.content)
            return _validate(result)
        except json.JSONDecodeError as exc:
            logger.warning("Risk agent JSON parse failed (attempt %d): %s", attempt + 1, exc)
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as exc:
            logger.error("Risk agent error (attempt %d): %s", attempt + 1, exc)
            if attempt < max_retries - 1:
                time.sleep(2)

    # Default safe: allow trade at 50% size if LLM unavailable
    logger.warning("Risk agent fallback: allowing 50% size")
    return {"veto": False, "reason": "LLM unavailable, reduced size", "adjusted_size_pct": 0.5}


def _validate(result: dict) -> dict:
    veto = bool(result.get("veto", False))
    size = float(result.get("adjusted_size_pct", 1.0))
    size = max(0.0, min(1.0, size))
    if veto:
        size = 0.0
    return {
        "veto": veto,
        "reason": str(result.get("reason", ""))[:100],
        "adjusted_size_pct": size,
    }
