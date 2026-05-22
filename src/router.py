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

    # Blend: RL position target + LLM directional bias
    llm_numeric = tech_signal.get("signal_numeric", 0.0)   # -1, 0, 1
    sentiment_net = sentiment.get("net_sentiment", 0.0)    # -1 to 1

    # Weighted blend: 60% RL, 25% tech LLM, 15% sentiment
    blended = 0.60 * rl_action + 0.25 * llm_numeric * llm_confidence + 0.15 * sentiment_net * 0.5
    blended = float(np.clip(blended, -1.0, 1.0))

    # Apply risk manager's size reduction
    size_pct = risk.get("adjusted_size_pct", 1.0)
    blended *= size_pct

    # Convert to integer units
    base_units = config.get("trading", {}).get("units", 10)
    target_units = int(round(blended * base_units))

    # Threshold: don't trade if change is tiny
    current_units = current_position.get("units", 0)
    threshold = config.get("risk", {}).get("position_change_threshold", 0.15)
    if abs(target_units - current_units) < (base_units * threshold):
        return _decision(
            current_units, "hold",
            f"Position change below threshold ({target_units} vs {current_units})",
            tech_signal, rl_action, llm_confidence,
        )

    action = "buy" if target_units > 0 else ("sell" if target_units < 0 else "close")
    reason = (f"RL={rl_action:.2f}, LLM={llm_signal}@{llm_confidence:.2f}, "
              f"sentiment={sentiment_net:.2f}, blended={blended:.2f}")

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
