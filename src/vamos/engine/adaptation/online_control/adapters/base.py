from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vamos.foundation.constraints.utils import compute_violation, is_feasible
from vamos.foundation.quality_indicators.pareto import pareto_filter

from ..contracts import HierarchicalAction, OperatorFamily


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def available_real_families() -> tuple[OperatorFamily, ...]:
    return (OperatorFamily.SBX_LIKE, OperatorFamily.DE_LIKE)


def summarize_constraints(G: np.ndarray | None) -> tuple[bool, float, float]:
    if G is None:
        return False, 1.0, 0.0
    feasible = is_feasible(G)
    violation = compute_violation(G)
    feasible_ratio = float(feasible.mean()) if feasible.size else 0.0
    mean_violation = float(violation.mean()) if violation.size else 0.0
    return True, feasible_ratio, mean_violation


def normalized_population_diversity(X: np.ndarray, xl: np.ndarray, xu: np.ndarray) -> float:
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2 or X_arr.shape[0] == 0:
        return 0.0
    span = np.asarray(xu, dtype=float) - np.asarray(xl, dtype=float)
    span_safe = np.where(np.abs(span) > 1e-12, span, 1.0)
    spread = np.std(X_arr, axis=0)
    return clamp01(float(np.mean(np.abs(spread / span_safe))))


def normalized_objective_extent(F: np.ndarray) -> float:
    F_arr = np.asarray(F, dtype=float)
    if F_arr.ndim != 2 or F_arr.shape[0] == 0:
        return 0.0
    extent = np.ptp(F_arr, axis=0)
    scale = np.max(np.abs(F_arr), axis=0)
    normalized = extent / (extent + scale + 1e-12)
    return clamp01(float(np.mean(normalized)))


def scalar_quality_indicator(F: np.ndarray, G: np.ndarray | None = None) -> float:
    F_arr = np.asarray(F, dtype=float)
    if F_arr.ndim != 2 or F_arr.shape[0] == 0:
        return 0.0
    if G is not None:
        feasible = is_feasible(G)
        if np.any(feasible):
            F_arr = F_arr[feasible]
    best = np.min(F_arr, axis=0)
    return float(-np.mean(best))


def nondominated_size(F: np.ndarray, G: np.ndarray | None = None) -> int:
    F_arr = np.asarray(F, dtype=float)
    if F_arr.ndim != 2 or F_arr.shape[0] == 0:
        return 0
    if G is not None:
        feasible = is_feasible(G)
        if np.any(feasible):
            F_arr = F_arr[feasible]
        else:
            return 0
    front = pareto_filter(F_arr, return_indices=False)
    if front is None:
        return 0
    return int(np.asarray(front).shape[0])


def budget_progress(evaluations: int, evaluation_budget: int | None) -> float:
    if evaluation_budget is None or evaluation_budget <= 0:
        return 0.0
    return clamp01(float(evaluations) / float(evaluation_budget))


def normalized_stagnation(stagnant_steps: int) -> float:
    return clamp01(float(stagnant_steps) / 5.0)


def update_quality_stagnation(
    *,
    best_quality: float | None,
    stagnant_steps: int,
    current_quality: float,
    tolerance: float = 1e-9,
) -> tuple[float, int]:
    if best_quality is None or current_quality > best_quality + tolerance:
        return current_quality, 0
    return best_quality, stagnant_steps + 1


@dataclass(frozen=True)
class VariationDescriptor:
    operator_family: OperatorFamily
    cross_method: str
    cross_params: Mapping[str, Any]
    mut_method: str
    mut_params: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator_family": self.operator_family.value,
            "cross_method": self.cross_method,
            "cross_params": {str(key): _to_plain_value(value) for key, value in self.cross_params.items()},
            "mut_method": self.mut_method,
            "mut_params": {str(key): _to_plain_value(value) for key, value in self.mut_params.items()},
            "metadata": {str(key): _to_plain_value(value) for key, value in self.metadata.items()},
        }


@dataclass(frozen=True)
class DecodedMOEADVariation:
    descriptor: VariationDescriptor
    crossover_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray]
    mutation_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray]
    cross_is_de: bool


def semantic_variation_descriptor(
    action: HierarchicalAction,
    *,
    n_var: int,
    metadata: Mapping[str, Any] | None = None,
) -> VariationDescriptor:
    intent = action.parametric_intent
    if action.operator_family is OperatorFamily.DE_LIKE:
        descriptor = _de_descriptor(intent, n_var=n_var)
    else:
        descriptor = _sbx_descriptor(intent, n_var=n_var)
    payload = {
        "regime": action.regime.value,
        "intent_prototype": action.parametric_intent.prototype,
        **descriptor.metadata,
        **dict(metadata or {}),
    }
    return VariationDescriptor(
        operator_family=descriptor.operator_family,
        cross_method=descriptor.cross_method,
        cross_params=descriptor.cross_params,
        mut_method=descriptor.mut_method,
        mut_params=descriptor.mut_params,
        metadata=payload,
    )


def _sbx_descriptor(intent: Any, *, n_var: int) -> VariationDescriptor:
    exploration = clamp01(intent.exploration_strength - 0.10 * intent.feasibility_bias)
    locality = clamp01(intent.locality + 0.15 * intent.feasibility_bias)
    mutation_strength = clamp01(intent.mutation_strength)
    crossover_prob = clamp01(0.78 + 0.20 * exploration - 0.08 * intent.feasibility_bias)
    eta_c = 3.0 + 47.0 * clamp01(0.65 * locality + 0.35 * (1.0 - exploration))
    prob_var = clamp01(0.25 + 0.65 * (0.65 * exploration + 0.35 * (1.0 - locality)))
    mutation_prob = _mutation_probability(intent, n_var=n_var, floor=0.45, span=1.60)
    eta_m = 4.0 + 46.0 * clamp01(0.60 * locality + 0.40 * (1.0 - mutation_strength))
    return VariationDescriptor(
        operator_family=OperatorFamily.SBX_LIKE,
        cross_method="sbx",
        cross_params={"prob": crossover_prob, "eta": eta_c, "prob_var": prob_var},
        mut_method="polynomial",
        mut_params={"prob": mutation_prob, "eta": eta_m},
        metadata={"decoder_family": "sbx_like"},
    )


def _de_descriptor(intent: Any, *, n_var: int) -> VariationDescriptor:
    exploration = clamp01(intent.exploration_strength)
    locality = clamp01(intent.locality + 0.10 * intent.feasibility_bias)
    mutation_strength = clamp01(intent.mutation_strength)
    f_value = clamp01(0.25 + 0.70 * clamp01(0.70 * exploration + 0.30 * mutation_strength) * (1.0 - 0.20 * locality))
    cr_value = clamp01(0.15 + 0.75 * clamp01(0.65 * exploration + 0.35 * (1.0 - locality)))
    mutation_prob = _mutation_probability(intent, n_var=n_var, floor=0.35, span=1.35)
    eta_m = 4.0 + 36.0 * clamp01(0.45 * locality + 0.55 * (1.0 - mutation_strength))
    return VariationDescriptor(
        operator_family=OperatorFamily.DE_LIKE,
        cross_method="de",
        cross_params={"F": f_value, "CR": cr_value},
        mut_method="polynomial",
        mut_params={"prob": mutation_prob, "eta": eta_m},
        metadata={"decoder_family": "de_like"},
    )


def _mutation_probability(intent: Any, *, n_var: int, floor: float, span: float) -> float:
    if n_var <= 0:
        return 0.1
    scaled = floor + span * clamp01(0.65 * intent.mutation_strength + 0.20 * intent.exploration_strength + 0.15 * intent.feasibility_bias)
    return min(1.0, (1.0 / float(n_var)) * scaled)


def _to_plain_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_value(item) for item in value]
    return value


def count_matching_rows(survivors: np.ndarray, candidates: np.ndarray) -> int:
    if survivors.size == 0 or candidates.size == 0:
        return 0
    matches = np.all(np.isclose(survivors[:, None, :], candidates[None, :, :], atol=1e-12, rtol=0.0), axis=2)
    return int(np.any(matches, axis=0).sum())


__all__ = [
    "DecodedMOEADVariation",
    "VariationDescriptor",
    "available_real_families",
    "budget_progress",
    "clamp01",
    "clamp_signed",
    "count_matching_rows",
    "nondominated_size",
    "normalized_objective_extent",
    "normalized_population_diversity",
    "normalized_stagnation",
    "scalar_quality_indicator",
    "semantic_variation_descriptor",
    "summarize_constraints",
    "update_quality_stagnation",
]
