"""
Signal Router: fuses LLM signals + RL policy decision → final trade order.

Output is a CONTINUOUS `target_size` in [-1, 1] (negative = short, positive = long).
Scheduler scales that by MAX_POSITION_OZ to compute actual oz to hold.
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

_BEST_MODEL_PATH = str(PROJECT_ROOT / "models" / "best_policy")
_BASELINE_MODEL_PATH = str(PROJECT_ROOT / "models" / "baseline_ppo")

# Maximum gross position in oz. target_size=1.0 ⇒ +MAX_POSITION_OZ long.
MAX_POSITION_OZ = 1.0

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
    Fuse LLM signals + RL policy into a continuous position decision.

    Returns:
        {
            "target_size": float,     # desired position normalized to [-1, 1]
            "action": str,            # "buy" | "sell" | "hold" | "close"
            "reason": str,
            "llm_signal": str,
            "rl_action": float,
            "confidence": float,
        }
    """
    current_units = float(current_position.get("units", 0.0))
    current_size_norm = float(np.clip(current_units / MAX_POSITION_OZ, -1.0, 1.0))

    # Risk veto → flatten
    if risk.get("veto", False):
        return _decision(0.0, current_size_norm, "close" if current_units != 0 else "hold",
                         f"Risk veto: {risk.get('reason', '')}", tech_signal, 0.0, 0.0, config)

    llm_signal = tech_signal.get("signal", "hold")
    llm_confidence = tech_signal.get("confidence", 0.0)
    min_confidence = config.get("risk", {}).get("min_llm_confidence", 0.6)

    if llm_confidence < min_confidence:
        return _decision(current_size_norm, current_size_norm, "hold",
                         f"LLM confidence {llm_confidence:.2f} < {min_confidence}",
                         tech_signal, 0.0, llm_confidence, config)

    rl_action = _get_rl_action(df_m5, tech_signal, sentiment, current_position, account_summary)

    llm_numeric  = tech_signal.get("signal_numeric", 0.0)
    llm_size     = tech_signal.get("size", llm_confidence)
    sentiment_net = sentiment.get("net_sentiment", 0.0)

    # Direction blend: 60% RL + 25% LLM + 15% sentiment.
    direction = 0.60 * rl_action + 0.25 * llm_numeric + 0.15 * (sentiment_net * 0.5)
    direction = float(np.clip(direction, -1.0, 1.0))

    # Size blend: 50% LLM declared size + 50% |RL action|. RL drives variance now.
    blended_size = 0.50 * llm_size + 0.50 * abs(rl_action)
    blended_size = float(np.clip(blended_size, 0.0, 1.0))

    # Risk-manager size adjustment (0.3 floor when not vetoed, to keep some skin in)
    size_pct = float(risk.get("adjusted_size_pct", 1.0))
    if size_pct < 0.3:
        size_pct = 0.3
    blended_size *= size_pct

    MIN_SIGNAL = 0.02
    if direction > MIN_SIGNAL:
        target_size = blended_size
    elif direction < -MIN_SIGNAL:
        target_size = -blended_size
    else:
        target_size = 0.0
    target_size = float(np.clip(target_size, -1.0, 1.0))

    logger.info(
        "Router: rl=%+.3f llm=%s(conf=%.2f size=%.2f) sent=%+.2f → dir=%+.3f size=%.2f → target=%+.3f (cur=%+.3f)",
        rl_action, llm_signal, llm_confidence, llm_size,
        sentiment_net, direction, blended_size, target_size, current_size_norm,
    )

    reason = (f"RL={rl_action:+.2f}, LLM={llm_signal}@{llm_confidence:.2f} size={llm_size:.2f}, "
              f"sent={sentiment_net:+.2f}, dir={direction:+.3f}")

    return _decision(target_size, current_size_norm, None, reason,
                     tech_signal, rl_action, llm_confidence, config)


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
        env.step_idx = max(0, len(env.df) - 1)
        obs = env._get_obs()
        obs[10] = float(tech_signal.get("signal_numeric", 0.0))
        obs[11] = float(tech_signal.get("confidence", 0.5))
        obs[12] = float((sentiment.get("net_sentiment", 0.0) + 1.0) / 2.0)
        units = float(current_position.get("units", 0.0))
        obs[13] = float(np.clip(units / MAX_POSITION_OZ, -1.0, 1.0))

        action, _ = policy.predict(obs, deterministic=True)
        return float(action[0])
    except Exception as exc:
        logger.warning("RL action failed: %s", exc)
        return 0.0


def _decision(
    target_size: float,
    current_size_norm: float,
    forced_action: Optional[str],
    reason: str,
    tech_signal: dict,
    rl_action: float,
    confidence: float,
    config: dict,
) -> dict:
    """Build the final decision dict, picking action based on the delta to current position."""
    if forced_action is not None:
        action = forced_action
    else:
        # Configurable hysteresis: skip orders whose size change is below the threshold.
        threshold = float(config.get("risk", {}).get("position_change_threshold", 0.15))
        delta = target_size - current_size_norm
        if abs(delta) < threshold:
            action = "hold"
        elif abs(target_size) < threshold:
            # Going (close to) flat → explicit close
            action = "close"
        elif target_size > 0:
            action = "buy"
        else:
            action = "sell"

    return {
        "target_size":   target_size,
        "current_size":  current_size_norm,
        "action":        action,
        "reason":        reason,
        "llm_signal":    tech_signal.get("signal", "hold"),
        "llm_reasoning": tech_signal.get("reasoning", ""),
        "rl_action":     rl_action,
        "confidence":    confidence,
    }
