"""
Reward helpers for adaptive operator selection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def survival_rate(survivors: int, offspring: int) -> float:
    """
    Fraction of offspring that survive.
    """
    if offspring <= 0:
        return 0.0
    return _clamp01(survivors / float(offspring))


def nd_insertion_rate(insertions: int, offspring: int) -> float:
    """
    Fraction of offspring inserted into the non-dominated set.
    """
    if offspring <= 0:
        return 0.0
    return _clamp01(insertions / float(offspring))


@dataclass(frozen=True)
class RewardSummary:
    """
    Aggregate reward with component breakdowns.
    """

    reward: float
    reward_survival: float
    reward_nd_insertions: float
    reward_hv_delta: float


def _normalize_weights(raw: Mapping[str, Any] | None) -> tuple[float, float, float]:
    if not raw:
        return 0.5, 0.5, 0.0
    w_surv = float(raw.get("survival", 0.0))
    w_nd = float(raw.get("nd_insertions", 0.0))
    w_hv = float(raw.get("hv_delta", 0.0))
    total = w_surv + w_nd + w_hv
    if total <= 0.0:
        return 0.5, 0.5, 0.0
    return w_surv / total, w_nd / total, w_hv / total


def aggregate_reward(
    reward_scope: str,
    survival_rate: float,
    nd_rate: float,
    hv_delta_rate: float = 0.0,
    weights: Mapping[str, Any] | None = None,
) -> RewardSummary:
    """
    Aggregate reward from individual components.
    """
    surv = _clamp01(survival_rate)
    nd = _clamp01(nd_rate)
    hv = _clamp01(hv_delta_rate)
    scope = (reward_scope or "combined").lower()

    if scope in {"survival", "survival_rate"}:
        reward = surv
    elif scope in {"nd", "nd_insertion", "nd_insertions"}:
        reward = nd
    elif scope in {"hv", "hv_delta", "hypervolume"}:
        reward = hv
    else:
        w_surv, w_nd, w_hv = _normalize_weights(weights)
        reward = w_surv * surv + w_nd * nd + w_hv * hv

    return RewardSummary(
        reward=_clamp01(reward),
        reward_survival=surv,
        reward_nd_insertions=nd,
        reward_hv_delta=hv,
    )


__all__ = [
    "RewardSummary",
    "aggregate_reward",
    "nd_insertion_rate",
    "survival_rate",
]
