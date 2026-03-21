from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class Regime(str, Enum):
    REPAIR = "repair"
    EXPAND = "expand"
    REFINE = "refine"


class OperatorFamily(str, Enum):
    SBX_LIKE = "sbx_like"
    DE_LIKE = "de_like"


@dataclass(frozen=True)
class SearchState:
    host: str
    step_index: int
    generation: int
    evaluations: int
    population_size: int
    objective_count: int
    decision_dim: int
    evaluation_budget: int | None = None
    budget_progress: float = 0.0
    is_constrained: bool = False
    feasible_ratio: float = 1.0
    mean_constraint_violation: float = 0.0
    extent: float = 0.0
    diversity: float = 0.0
    stagnation: float = 0.0
    quality_indicator: float = 0.0
    available_families: tuple[OperatorFamily, ...] = (OperatorFamily.SBX_LIKE,)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "step_index": self.step_index,
            "generation": self.generation,
            "evaluations": self.evaluations,
            "population_size": self.population_size,
            "objective_count": self.objective_count,
            "decision_dim": self.decision_dim,
            "evaluation_budget": self.evaluation_budget,
            "budget_progress": self.budget_progress,
            "is_constrained": self.is_constrained,
            "feasible_ratio": self.feasible_ratio,
            "mean_constraint_violation": self.mean_constraint_violation,
            "extent": self.extent,
            "diversity": self.diversity,
            "stagnation": self.stagnation,
            "quality_indicator": self.quality_indicator,
            "available_families": [family.value for family in self.available_families],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ParametricIntent:
    prototype: str | None = None
    exploration_strength: float = 0.5
    locality: float = 0.5
    mutation_strength: float = 0.5
    feasibility_bias: float = 0.5
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prototype": self.prototype,
            "exploration_strength": self.exploration_strength,
            "locality": self.locality,
            "mutation_strength": self.mutation_strength,
            "feasibility_bias": self.feasibility_bias,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class HierarchicalAction:
    regime: Regime
    operator_family: OperatorFamily
    parametric_intent: ParametricIntent = field(default_factory=ParametricIntent)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value,
            "operator_family": self.operator_family.value,
            "parametric_intent": self.parametric_intent.as_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Outcome:
    success: bool
    reward_hint: float = 0.0
    survivor_ratio: float = 0.0
    accepted_ratio: float | None = None
    nd_insertions_ratio: float | None = None
    feasible_ratio: float | None = None
    feasible_delta: float | None = None
    extent: float | None = None
    extent_gain: float | None = None
    diversity: float | None = None
    diversity_delta: float | None = None
    diversity_gain: float | None = None
    quality_indicator: float | None = None
    quality_delta: float | None = None
    overhead_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "reward_hint": self.reward_hint,
            "survivor_ratio": self.survivor_ratio,
            "accepted_ratio": self.accepted_ratio,
            "nd_insertions_ratio": self.nd_insertions_ratio,
            "feasible_ratio": self.feasible_ratio,
            "feasible_delta": self.feasible_delta,
            "extent": self.extent,
            "extent_gain": self.extent_gain,
            "diversity": self.diversity,
            "diversity_delta": self.diversity_delta,
            "diversity_gain": self.diversity_gain,
            "quality_indicator": self.quality_indicator,
            "quality_delta": self.quality_delta,
            "overhead_ms": self.overhead_ms,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Credit:
    reward: float
    bounded_reward: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "bounded_reward": self.bounded_reward,
            "metadata": dict(self.metadata),
        }


class HostAdapter(Protocol):
    host_name: str

    def build_search_state(self, host_state: Any) -> SearchState: ...

    def decode_action(self, action: HierarchicalAction, host_state: Any) -> Any: ...

    def build_outcome(
        self,
        *,
        before: SearchState,
        after: SearchState,
        action: HierarchicalAction,
        host_state: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> Outcome: ...


class RegimeRouter(Protocol):
    def route(self, search_state: SearchState) -> Regime: ...

    def update(
        self,
        search_state: SearchState,
        action: HierarchicalAction,
        outcome: Outcome,
        credit: Credit,
    ) -> None: ...


class HierarchicalPolicy(Protocol):
    def select_action(self, search_state: SearchState, regime: Regime) -> HierarchicalAction: ...

    def update(
        self,
        search_state: SearchState,
        action: HierarchicalAction,
        outcome: Outcome,
        credit: Credit,
    ) -> None: ...


class CreditModel(Protocol):
    def compute(self, search_state: SearchState, action: HierarchicalAction, outcome: Outcome) -> Credit: ...


__all__ = [
    "Credit",
    "CreditModel",
    "HierarchicalAction",
    "HierarchicalPolicy",
    "HostAdapter",
    "OperatorFamily",
    "Outcome",
    "ParametricIntent",
    "Regime",
    "RegimeRouter",
    "SearchState",
]
