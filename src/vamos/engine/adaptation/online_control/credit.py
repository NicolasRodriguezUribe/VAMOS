from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import Credit, CreditModel, HierarchicalAction, Outcome, SearchState


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _available_component(value: float | None) -> bool:
    return value is not None


def _positive_gain(value: float | None) -> float | None:
    if value is None:
        return None
    return _clamp01(max(0.0, float(value)))


def _component_map(outcome: Outcome) -> dict[str, float]:
    components: dict[str, float] = {}
    accepted = outcome.accepted_ratio
    if accepted is None and outcome.survivor_ratio > 0.0:
        accepted = outcome.survivor_ratio
    if accepted is not None:
        components["accepted_ratio"] = _clamp01(float(accepted))
    if _available_component(outcome.nd_insertions_ratio):
        components["nd_insertions_ratio"] = _clamp01(float(outcome.nd_insertions_ratio))
    feasible_gain = _positive_gain(outcome.feasible_delta)
    if feasible_gain is not None:
        components["feasible_delta"] = feasible_gain
    extent_gain = _positive_gain(outcome.extent_gain)
    if extent_gain is not None:
        components["extent_gain"] = extent_gain
    diversity_gain = _positive_gain(outcome.diversity_gain if outcome.diversity_gain is not None else outcome.diversity_delta)
    if diversity_gain is not None:
        components["diversity_gain"] = diversity_gain
    return components


def _weighted_credit(
    outcome: Outcome,
    *,
    weights: Mapping[str, float],
) -> tuple[float, dict[str, float], float]:
    components = _component_map(outcome)
    if not components:
        fallback = _clamp01(outcome.reward_hint)
        return fallback, {}, 0.0

    active_weight = 0.0
    total = 0.0
    for key, weight in weights.items():
        if key not in components:
            continue
        w = float(weight)
        active_weight += w
        total += w * components[key]
    if active_weight <= 0.0:
        fallback = _clamp01(outcome.reward_hint)
        return fallback, components, 0.0
    reward = _clamp01(total / active_weight)
    return reward, components, active_weight


@dataclass
class NoOpCreditModel(CreditModel):
    default_reward: float = 0.0

    def compute(self, search_state: SearchState, action: HierarchicalAction, outcome: Outcome) -> Credit:
        del search_state, action
        bounded = _clamp01(outcome.reward_hint if outcome is not None else self.default_reward)
        return Credit(
            reward=bounded,
            bounded_reward=bounded,
            metadata={"model": "noop"},
        )


@dataclass
class SimpleImprovementCreditModel(CreditModel):
    """Bounded transparent reward based on acceptance and population improvement signals."""

    accepted_ratio_weight: float = 0.35
    nd_insertions_weight: float = 0.25
    feasible_delta_weight: float = 0.15
    extent_gain_weight: float = 0.15
    diversity_gain_weight: float = 0.10

    def compute(self, search_state: SearchState, action: HierarchicalAction, outcome: Outcome) -> Credit:
        del search_state, action
        weights = {
            "accepted_ratio": self.accepted_ratio_weight,
            "nd_insertions_ratio": self.nd_insertions_weight,
            "feasible_delta": self.feasible_delta_weight,
            "extent_gain": self.extent_gain_weight,
            "diversity_gain": self.diversity_gain_weight,
        }
        reward, components, active_weight = _weighted_credit(outcome, weights=weights)
        return Credit(
            reward=reward,
            bounded_reward=reward,
            metadata={
                "model": "simple_improvement",
                "components": components,
                "active_weight": active_weight,
            },
        )


@dataclass
class CostAwareCreditModel(SimpleImprovementCreditModel):
    """Improvement credit with a gentle bounded penalty for control overhead."""

    overhead_scale_ms: float = 5.0
    max_penalty: float = 0.15

    def compute(self, search_state: SearchState, action: HierarchicalAction, outcome: Outcome) -> Credit:
        base_credit = super().compute(search_state, action, outcome)
        overhead_ms = outcome.overhead_ms
        penalty = 0.0
        if overhead_ms is not None and overhead_ms > 0.0:
            normalized = float(overhead_ms) / (float(overhead_ms) + max(1e-9, self.overhead_scale_ms))
            penalty = self.max_penalty * _clamp01(normalized)
        reward = _clamp01(base_credit.bounded_reward - penalty)
        metadata: dict[str, Any] = dict(base_credit.metadata)
        metadata.update(
            {
                "model": "cost_aware",
                "overhead_ms": overhead_ms,
                "overhead_penalty": penalty,
            }
        )
        return Credit(
            reward=reward,
            bounded_reward=reward,
            metadata=metadata,
        )


__all__ = [
    "CostAwareCreditModel",
    "NoOpCreditModel",
    "SimpleImprovementCreditModel",
]
