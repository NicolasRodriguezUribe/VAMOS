from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .contracts import Credit, HierarchicalAction, HierarchicalPolicy, OperatorFamily, Outcome, Regime, SearchState
from .prototypes import DEFAULT_PROTOTYPE_SET, available_intent_prototypes, build_intent_prototype, normalize_prototype_set


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _available_families(search_state: SearchState) -> tuple[OperatorFamily, ...]:
    return search_state.available_families or (OperatorFamily.SBX_LIKE,)


def _resolve_family(preferred: OperatorFamily, search_state: SearchState) -> OperatorFamily:
    families = _available_families(search_state)
    return preferred if preferred in families else families[0]


def _ordered_prototypes(
    prototype_names: Sequence[str] | None,
    *,
    prototype_set: str,
) -> tuple[str, ...]:
    if prototype_names is None:
        return available_intent_prototypes(prototype_set)
    return tuple(str(name).strip().lower() for name in prototype_names)


def _heuristic_family(search_state: SearchState, regime: Regime) -> OperatorFamily:
    families = _available_families(search_state)
    if regime is Regime.REPAIR:
        return _resolve_family(OperatorFamily.SBX_LIKE, search_state)
    if regime is Regime.EXPAND and OperatorFamily.DE_LIKE in families:
        if (
            search_state.budget_progress < 0.72
            or search_state.diversity < 0.24
            or search_state.extent < 0.22
            or search_state.stagnation > 0.35
        ):
            return OperatorFamily.DE_LIKE
    return _resolve_family(OperatorFamily.SBX_LIKE, search_state)


def _heuristic_prototype(search_state: SearchState, regime: Regime, family: OperatorFamily) -> str:
    if regime is Regime.REPAIR:
        return "feasibility_biased"
    if search_state.is_constrained and search_state.feasible_ratio < 0.65:
        return "feasibility_biased"
    if regime is Regime.REFINE:
        return "local_refine"
    if search_state.stagnation >= 0.65:
        return "mutation_heavy"
    if regime is Regime.EXPAND and (
        search_state.diversity < 0.22
        or search_state.extent < 0.22
        or search_state.budget_progress < 0.35
    ):
        return "exploratory"
    if family is OperatorFamily.DE_LIKE and search_state.stagnation >= 0.40:
        return "mutation_heavy"
    return "balanced"


def _heuristic_family_bias(search_state: SearchState, regime: Regime, family: OperatorFamily) -> float:
    score = 0.0
    if family not in _available_families(search_state):
        return -1.0
    if regime is Regime.REPAIR:
        score += 0.18 if family is OperatorFamily.SBX_LIKE else -0.18
    elif regime is Regime.EXPAND:
        if family is OperatorFamily.DE_LIKE:
            score += 0.08
            if search_state.diversity < 0.24 or search_state.extent < 0.22:
                score += 0.06
            if search_state.stagnation > 0.40:
                score += 0.05
        else:
            score += 0.03
    else:
        score += 0.14 if family is OperatorFamily.SBX_LIKE else -0.10
    return score


def _heuristic_prototype_bias(search_state: SearchState, regime: Regime, family: OperatorFamily, prototype: str) -> float:
    score = 0.0
    if regime is Regime.REPAIR:
        score += 0.18 if prototype == "feasibility_biased" else -0.08
    elif regime is Regime.REFINE:
        if prototype == "local_refine":
            score += 0.16
        elif prototype == "balanced":
            score += 0.04
        else:
            score -= 0.05
    else:
        if search_state.stagnation >= 0.65 and prototype == "mutation_heavy":
            score += 0.14
        if (search_state.diversity < 0.22 or search_state.extent < 0.22) and prototype == "exploratory":
            score += 0.14
        if search_state.budget_progress < 0.35 and prototype == "exploratory":
            score += 0.08
        if prototype == "balanced":
            score += 0.02
    if search_state.is_constrained and search_state.feasible_ratio < 0.65 and prototype == "feasibility_biased":
        score += 0.10
    if family is OperatorFamily.DE_LIKE and prototype == "exploratory":
        score += 0.03
    if family is OperatorFamily.SBX_LIKE and prototype == "local_refine":
        score += 0.03
    return score


@dataclass
class _SemanticScoreTracker:
    exploration_weight: float = 0.30
    counts: dict[str, int] = field(default_factory=dict)
    reward_sums: dict[str, float] = field(default_factory=dict)

    def select(self, options: Sequence[str], *, biases: dict[str, float] | None = None) -> str:
        if not options:
            raise ValueError("Adaptive policy requires at least one semantic option.")
        bias_map = biases or {}
        ordered = [str(option) for option in options]
        total_updates = max(1, sum(self.counts.values()))
        best_option = ordered[0]
        best_score = float("-inf")
        for option in ordered:
            count = self.counts.get(option, 0)
            bias = float(bias_map.get(option, 0.0))
            if count <= 0:
                score = 1.0 + bias
            else:
                mean_reward = self.reward_sums.get(option, 0.0) / float(count)
                score = mean_reward + self.exploration_weight * math.sqrt(math.log(total_updates + 1.0) / float(count)) + bias
            if score > best_score:
                best_score = score
                best_option = option
        return best_option

    def update(self, option: str, reward: float) -> None:
        key = str(option)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.reward_sums[key] = self.reward_sums.get(key, 0.0) + _clamp01(reward)

    def summary(self, option: str) -> dict[str, float]:
        key = str(option)
        count = self.counts.get(key, 0)
        total = self.reward_sums.get(key, 0.0)
        mean_reward = total / float(count) if count > 0 else 0.0
        return {"count": float(count), "mean_reward": mean_reward}

    def export_state(self) -> dict[str, object]:
        return {
            "exploration_weight": self.exploration_weight,
            "counts": {str(key): int(value) for key, value in self.counts.items()},
            "reward_sums": {str(key): float(value) for key, value in self.reward_sums.items()},
        }

    def load_state(self, state: Mapping[str, object]) -> None:
        weight = state.get("exploration_weight", self.exploration_weight)
        self.exploration_weight = float(weight) if isinstance(weight, (int, float)) else self.exploration_weight
        counts = state.get("counts", {})
        reward_sums = state.get("reward_sums", {})
        if isinstance(counts, Mapping):
            self.counts = {str(key): int(value) for key, value in counts.items()}
        if isinstance(reward_sums, Mapping):
            self.reward_sums = {str(key): float(value) for key, value in reward_sums.items()}


def _build_action(
    *,
    regime: Regime,
    family: OperatorFamily,
    prototype: str,
    policy_name: str,
    fixed_family: OperatorFamily | None = None,
    selection_mode: str,
    family_summary: dict[str, float] | None = None,
    intent_summary: dict[str, float] | None = None,
    prototype_set: str = DEFAULT_PROTOTYPE_SET,
) -> HierarchicalAction:
    intent = build_intent_prototype(prototype, prototype_set=prototype_set)
    metadata: dict[str, object] = {
        "policy": policy_name,
        "selection_mode": selection_mode,
        "intent_prototype": prototype,
    }
    if fixed_family is not None:
        metadata["fixed_family"] = fixed_family.value
    if family_summary is not None:
        metadata["family_evidence_count"] = int(family_summary["count"])
        metadata["family_mean_reward"] = float(family_summary["mean_reward"])
    if intent_summary is not None:
        metadata["intent_evidence_count"] = int(intent_summary["count"])
        metadata["intent_mean_reward"] = float(intent_summary["mean_reward"])
    return HierarchicalAction(
        regime=regime,
        operator_family=family,
        parametric_intent=intent,
        metadata=metadata,
    )


@dataclass
class FlatOperatorPolicy(HierarchicalPolicy):
    prototype_set: str = DEFAULT_PROTOTYPE_SET

    def __post_init__(self) -> None:
        self.prototype_set = normalize_prototype_set(self.prototype_set)

    def select_action(self, search_state: SearchState, regime: Regime) -> HierarchicalAction:
        family = _heuristic_family(search_state, regime)
        return _build_action(
            regime=regime,
            family=family,
            prototype="balanced",
            policy_name="flat_operator",
            selection_mode="heuristic",
            prototype_set=self.prototype_set,
        )

    def update(self, search_state: SearchState, action: HierarchicalAction, outcome: Outcome, credit: Credit) -> None:
        del search_state, action, outcome, credit

    def export_state(self) -> dict[str, object]:
        return {"policy": "flat_operator", "prototype_set": self.prototype_set}

    def load_state(self, state: Mapping[str, object]) -> None:
        del state


@dataclass
class FlatParameterPolicy(HierarchicalPolicy):
    fixed_family: OperatorFamily = OperatorFamily.SBX_LIKE
    prototype_set: str = DEFAULT_PROTOTYPE_SET

    def __post_init__(self) -> None:
        self.prototype_set = normalize_prototype_set(self.prototype_set)

    def select_action(self, search_state: SearchState, regime: Regime) -> HierarchicalAction:
        family = _resolve_family(self.fixed_family, search_state)
        prototype = _heuristic_prototype(search_state, regime, family)
        return _build_action(
            regime=regime,
            family=family,
            prototype=prototype,
            policy_name="flat_parameter",
            fixed_family=family,
            selection_mode="heuristic",
            prototype_set=self.prototype_set,
        )

    def update(self, search_state: SearchState, action: HierarchicalAction, outcome: Outcome, credit: Credit) -> None:
        del search_state, action, outcome, credit

    def export_state(self) -> dict[str, object]:
        return {
            "policy": "flat_parameter",
            "prototype_set": self.prototype_set,
            "fixed_family": self.fixed_family.value,
        }

    def load_state(self, state: Mapping[str, object]) -> None:
        del state


@dataclass
class HierarchicalJointPolicy(HierarchicalPolicy):
    prototype_set: str = DEFAULT_PROTOTYPE_SET

    def __post_init__(self) -> None:
        self.prototype_set = normalize_prototype_set(self.prototype_set)

    def select_action(self, search_state: SearchState, regime: Regime) -> HierarchicalAction:
        family = _heuristic_family(search_state, regime)
        prototype = _heuristic_prototype(search_state, regime, family)
        return _build_action(
            regime=regime,
            family=family,
            prototype=prototype,
            policy_name="hierarchical_joint",
            selection_mode="heuristic",
            prototype_set=self.prototype_set,
        )

    def update(self, search_state: SearchState, action: HierarchicalAction, outcome: Outcome, credit: Credit) -> None:
        del search_state, action, outcome, credit

    def export_state(self) -> dict[str, object]:
        return {"policy": "hierarchical_joint", "prototype_set": self.prototype_set}

    def load_state(self, state: Mapping[str, object]) -> None:
        del state


@dataclass
class AdaptiveFlatOperatorPolicy(HierarchicalPolicy):
    prototype_set: str = DEFAULT_PROTOTYPE_SET
    family_tracker: _SemanticScoreTracker = field(default_factory=_SemanticScoreTracker)

    def __post_init__(self) -> None:
        self.prototype_set = normalize_prototype_set(self.prototype_set)

    def select_action(self, search_state: SearchState, regime: Regime) -> HierarchicalAction:
        families = _available_families(search_state)
        labels = [family.value for family in families]
        biases = {family.value: _heuristic_family_bias(search_state, regime, family) for family in families}
        selected = OperatorFamily(self.family_tracker.select(labels, biases=biases))
        return _build_action(
            regime=regime,
            family=selected,
            prototype="balanced",
            policy_name="adaptive_flat_operator",
            selection_mode="adaptive_ucb",
            family_summary=self.family_tracker.summary(selected.value),
            prototype_set=self.prototype_set,
        )

    def update(self, search_state: SearchState, action: HierarchicalAction, outcome: Outcome, credit: Credit) -> None:
        del search_state, outcome
        self.family_tracker.update(action.operator_family.value, credit.bounded_reward)

    def export_state(self) -> dict[str, object]:
        return {
            "policy": "adaptive_flat_operator",
            "prototype_set": self.prototype_set,
            "family_tracker": self.family_tracker.export_state(),
        }

    def load_state(self, state: Mapping[str, object]) -> None:
        tracker_state = state.get("family_tracker")
        if isinstance(tracker_state, Mapping):
            self.family_tracker.load_state(tracker_state)


@dataclass
class AdaptiveFlatParameterPolicy(HierarchicalPolicy):
    fixed_family: OperatorFamily = OperatorFamily.SBX_LIKE
    prototype_set: str = DEFAULT_PROTOTYPE_SET
    prototype_names: tuple[str, ...] | None = None
    prototype_tracker: _SemanticScoreTracker = field(default_factory=_SemanticScoreTracker)

    def __post_init__(self) -> None:
        self.prototype_set = normalize_prototype_set(self.prototype_set)
        self.prototype_names = _ordered_prototypes(self.prototype_names, prototype_set=self.prototype_set)

    def select_action(self, search_state: SearchState, regime: Regime) -> HierarchicalAction:
        family = _resolve_family(self.fixed_family, search_state)
        biases = {
            prototype: _heuristic_prototype_bias(search_state, regime, family, prototype)
            for prototype in self.prototype_names or ()
        }
        prototype = self.prototype_tracker.select(self.prototype_names or (), biases=biases)
        return _build_action(
            regime=regime,
            family=family,
            prototype=prototype,
            policy_name="adaptive_flat_parameter",
            fixed_family=family,
            selection_mode="adaptive_ucb",
            intent_summary=self.prototype_tracker.summary(prototype),
            prototype_set=self.prototype_set,
        )

    def update(self, search_state: SearchState, action: HierarchicalAction, outcome: Outcome, credit: Credit) -> None:
        del search_state, outcome
        prototype = action.parametric_intent.prototype or "balanced"
        self.prototype_tracker.update(prototype, credit.bounded_reward)

    def export_state(self) -> dict[str, object]:
        return {
            "policy": "adaptive_flat_parameter",
            "prototype_set": self.prototype_set,
            "fixed_family": self.fixed_family.value,
            "prototype_names": list(self.prototype_names or ()),
            "prototype_tracker": self.prototype_tracker.export_state(),
        }

    def load_state(self, state: Mapping[str, object]) -> None:
        tracker_state = state.get("prototype_tracker")
        if isinstance(tracker_state, Mapping):
            self.prototype_tracker.load_state(tracker_state)


@dataclass
class AdaptiveHierarchicalJointPolicy(HierarchicalPolicy):
    prototype_set: str = DEFAULT_PROTOTYPE_SET
    prototype_names: tuple[str, ...] | None = None
    family_tracker: _SemanticScoreTracker = field(default_factory=_SemanticScoreTracker)
    prototype_trackers: dict[str, _SemanticScoreTracker] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prototype_set = normalize_prototype_set(self.prototype_set)
        self.prototype_names = _ordered_prototypes(self.prototype_names, prototype_set=self.prototype_set)

    def _prototype_tracker(self, family: OperatorFamily) -> _SemanticScoreTracker:
        key = family.value
        tracker = self.prototype_trackers.get(key)
        if tracker is None:
            tracker = _SemanticScoreTracker()
            self.prototype_trackers[key] = tracker
        return tracker

    def select_action(self, search_state: SearchState, regime: Regime) -> HierarchicalAction:
        families = _available_families(search_state)
        family_biases = {family.value: _heuristic_family_bias(search_state, regime, family) for family in families}
        family = OperatorFamily(self.family_tracker.select([family.value for family in families], biases=family_biases))
        tracker = self._prototype_tracker(family)
        prototype_biases = {
            prototype: _heuristic_prototype_bias(search_state, regime, family, prototype)
            for prototype in self.prototype_names or ()
        }
        prototype = tracker.select(self.prototype_names or (), biases=prototype_biases)
        return _build_action(
            regime=regime,
            family=family,
            prototype=prototype,
            policy_name="adaptive_hierarchical_joint",
            selection_mode="adaptive_ucb",
            family_summary=self.family_tracker.summary(family.value),
            intent_summary=tracker.summary(prototype),
            prototype_set=self.prototype_set,
        )

    def update(self, search_state: SearchState, action: HierarchicalAction, outcome: Outcome, credit: Credit) -> None:
        del search_state, outcome
        self.family_tracker.update(action.operator_family.value, credit.bounded_reward)
        prototype = action.parametric_intent.prototype or "balanced"
        self._prototype_tracker(action.operator_family).update(prototype, credit.bounded_reward)

    def export_state(self) -> dict[str, object]:
        return {
            "policy": "adaptive_hierarchical_joint",
            "prototype_set": self.prototype_set,
            "prototype_names": list(self.prototype_names or ()),
            "family_tracker": self.family_tracker.export_state(),
            "prototype_trackers": {
                key: tracker.export_state()
                for key, tracker in sorted(self.prototype_trackers.items())
            },
        }

    def load_state(self, state: Mapping[str, object]) -> None:
        family_state = state.get("family_tracker")
        if isinstance(family_state, Mapping):
            self.family_tracker.load_state(family_state)
        prototype_states = state.get("prototype_trackers")
        if isinstance(prototype_states, Mapping):
            for key, tracker_state in prototype_states.items():
                if not isinstance(tracker_state, Mapping):
                    continue
                tracker = self.prototype_trackers.get(str(key))
                if tracker is None:
                    tracker = _SemanticScoreTracker()
                    self.prototype_trackers[str(key)] = tracker
                tracker.load_state(tracker_state)


__all__ = [
    "AdaptiveFlatOperatorPolicy",
    "AdaptiveFlatParameterPolicy",
    "AdaptiveHierarchicalJointPolicy",
    "FlatOperatorPolicy",
    "FlatParameterPolicy",
    "HierarchicalJointPolicy",
]
