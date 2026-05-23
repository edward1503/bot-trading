"""
Signal Router: fuses LLM signals + RL policy decision → final trade order.
This is the brain of the 5-minute trading loop.
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)

_BEST_MODEL_PATH = "models/best_policy"
_BASELINE_MODEL_PATH = "models/baseline_ppo"

_policy: Optional[PPO] = None


def get_policy() -> Optional[PPO]:
    global _policy
    best_path = _BEST_MODEL_PATH + ".zip"
    baseline_path = _BASELINE_MODEL_PATH + ".zip"

    if os.path.exists(best_path):
        model_path = _BEST_MODEL_PATH
    elif os.path.exists(baseline_path):
        model_path = _BASELINE_MODEL_PATH
    else:
        logger.warning("No RL policy found, will use LLM signals only")
        return None

    if _policy is None:
        logger.info("Loading RL policy from %s", model_path)
        _policy = PPO.load(model_path)
    return _policy


def reload_policy():
    global _policy
    _policy = None
    return get_policy()


def decide(
    df_m5: pd.DataFrame,
    tech_signal: dict,
    sentiment: dict,
    risk: dict,
    current_position: dict,
    account_summary: dict,
    config: dict,
) -> dict:
    """
    Fuse LLM signals + RL policy into a final position decision.

    Returns:
        {
            "target_units": int,    # desired position in OANDA units
            "action": str,          # "buy" | "sell" | "hold" | "close"
            "reason": str,
            "llm_signal": str,
            "rl_action": float,
            "confidence": float,
        }
    """
    # Risk veto: abort immediately
    if risk.get("veto", False):
        return _decision(0, "hold", f"Risk veto: {risk.get('reason', '')}", tech_signal, 0.0, 0.0)

    llm_signal = tech_signal.get("signal", "hold")
    llm_confidence = tech_signal.get("confidence", 0.0)
    min_confidence = config.get("risk", {}).get("min_llm_confidence", 0.6)

    if llm_confidence < min_confidence:
        return _decision(
            current_position.get("units", 0), "hold",
            f"LLM confidence {llm_confidence:.2f} < {min_confidence}",
            tech_signal, 0.0, llm_confidence,
        )

    # RL policy: build observation with LLM signals injected
    rl_action = _get_rl_action(df_m5, tech_signal, sentiment, current_position, account_summary)

    # LLM-determined size (0.0–1.0) và direction (-1, 0, 1)
    llm_numeric  = tech_signal.get("signal_numeric", 0.0)  # -1 / 0 / 1
    llm_size     = tech_signal.get("size", llm_confidence) # LLM tự quyết khối lượng
    sentiment_net = sentiment.get("net_sentiment", 0.0)

    # Blend direction: 60% RL + 25% LLM + 15% sentiment
    direction = 0.60 * rl_action + 0.25 * llm_numeric + 0.15 * sentiment_net * 0.5
    direction = float(np.clip(direction, -1.0, 1.0))

    # Blend size: 50% LLM size + 50% |RL action| (RL cũng biết nên đặt bao nhiêu)
    blended_size = 0.50 * llm_size + 0.50 * abs(rl_action)
    blended_size = float(np.clip(blended_size, 0.0, 1.0))

    # Apply risk manager's size reduction
    size_pct = risk.get("adjusted_size_pct", 1.0)
    if not risk.get("veto", False) and size_pct < 0.3:
        size_pct = 0.3
    blended_size *= size_pct

    # Final target: direction × size → [-1, 1] float
    MIN_SIGNAL = 0.02
    if direction > MIN_SIGNAL:
        target_size_float = blended_size          # long
    elif direction < -MIN_SIGNAL:
        target_size_float = -blended_size         # short
    else:
        target_size_float = 0.0                   # flat

    # Convert to integer units (1 unit = 1 lệnh minimum)
    target_units = 1 if target_size_float > MIN_SIGNAL else (-1 if target_size_float < -MIN_SIGNAL else 0)

    logger.info(
        "Router: rl=%.3f llm=%s(conf=%.2f size=%.2f) sentiment=%.2f "
        "→ dir=%.3f size=%.2f → target=%+d",
        rl_action, llm_signal, llm_confidence, llm_size,
        sentiment_net, direction, blended_size, target_units,
    )

    # Nếu không thay đổi so với vị thế hiện tại thì hold
    current_units = current_position.get("units", 0)
    current_dir = 1 if current_units > 0 else (-1 if current_units < 0 else 0)
    if target_units == current_dir:
        return _decision(
            current_units, "hold",
            f"Already in target direction ({target_units:+d})",
            tech_signal, rl_action, llm_confidence,
        )

    action = "buy" if target_units > 0 else ("sell" if target_units < 0 else "close")
    reason = (f"RL={rl_action:.2f}, LLM={llm_signal}@{llm_confidence:.2f} size={llm_size:.2f}, "
              f"sentiment={sentiment_net:.2f}, dir={direction:.3f} size={blended_size:.2f}")

    return _decision(target_units, action, reason, tech_signal, rl_action, llm_confidence)


def _get_rl_action(
    df: pd.DataFrame,
    tech_signal: dict,
    sentiment: dict,
    current_position: dict,
    account_summary: dict,
) -> float:
    policy = get_policy()
    if policy is None or df.empty:
        return 0.0

    from src.rl.env import XAUUSDTradingEnv
    try:
        env = XAUUSDTradingEnv(df)
        obs, _ = env.reset()
        # Inject LLM signals into observation indices 10, 11, 12
        obs[10] = float(tech_signal.get("signal_numeric", 0.0))
        obs[11] = float(tech_signal.get("confidence", 0.5))
        obs[12] = float((sentiment.get("net_sentiment", 0.0) + 1.0) / 2.0)  # normalize to 0-1
        obs[13] = float(current_position.get("units", 0)) / 10.0

        action, _ = policy.predict(obs, deterministic=True)
        return float(action[0])
    except Exception as exc:
        logger.warning("RL action failed: %s", exc)
        return 0.0


def _decision(
    target_units: int,
    action: str,
    reason: str,
    tech_signal: dict,
    rl_action: float,
    confidence: float,
) -> dict:
    return {
        "target_units": target_units,
        "action": action,
        "reason": reason,
        "llm_signal": tech_signal.get("signal", "hold"),
        "llm_reasoning": tech_signal.get("reasoning", ""),
        "rl_action": rl_action,
        "confidence": confidence,
    }
