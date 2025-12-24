from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Sequence

import numpy as np

from vamos.ux.analysis.mcdm import weighted_sum_scores, tchebycheff_scores, knee_point_scores, reference_point_scores, topsis_scores
from vamos.ux.studio.data import FrontRecord, normalize_objectives


class SolutionDecoder(Protocol):
    def decode(self, x: np.ndarray) -> Dict[str, Any]:
        ...


def default_decoder(var: np.ndarray) -> Dict[str, Any]:
    return {f"x{i}": float(val) for i, val in enumerate(var)}


def build_decoder(_problem_name: str, metadata: dict | None = None) -> SolutionDecoder:
    # Placeholder for richer problem-specific decoding; metadata can be extended later.
    return default_decoder


def compute_mcdm_scores(
    F: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    reference_point: np.ndarray | None = None,
    methods: Sequence[str] = ("weighted_sum", "tchebycheff", "knee"),
) -> Dict[str, np.ndarray]:
    scores: Dict[str, np.ndarray] = {}
    if weights is None:
        weights = np.ones(F.shape[1]) / F.shape[1]
    for method in methods:
        if method == "weighted_sum":
            scores[method] = weighted_sum_scores(F, weights).scores
        elif method in {"tchebycheff", "tchebychev"}:
            scores[method] = tchebycheff_scores(F, weights, reference_point).scores
        elif method == "knee":
            try:
                scores[method] = knee_point_scores(F).scores
            except ValueError:
                continue
        elif method == "reference":
            if reference_point is not None:
                scores[method] = reference_point_scores(F, reference_point).scores
        elif method == "topsis":
             scores[method] = topsis_scores(F, weights).scores
        else:
            continue
    return scores


@dataclass
class DecisionView:
    front: FrontRecord
    normalized_F: np.ndarray
    decoded_X: List[Dict[str, Any]] | None
    constraints: np.ndarray | None
    mcdm_scores: Dict[str, np.ndarray] = field(default_factory=dict)


def build_decision_view(
    front: FrontRecord,
    *,
    decoder: SolutionDecoder | None = None,
    weights: np.ndarray | None = None,
    reference_point: np.ndarray | None = None,
    methods: Sequence[str] = ("weighted_sum", "tchebycheff", "knee"),
) -> DecisionView:
    F = front.points_F
    norm_F = normalize_objectives(F)
    decoded = None
    if front.points_X is not None:
        dec = decoder or build_decoder(front.problem_name, front.extra.get("metadata", {}))
        decoded = [dec(x) for x in front.points_X]
    scores = compute_mcdm_scores(norm_F, weights=weights, reference_point=reference_point, methods=methods)
    return DecisionView(
        front=front,
        normalized_F=norm_F,
        decoded_X=decoded,
        constraints=front.constraints,
        mcdm_scores=scores,
    )


def rank_by_score(view: DecisionView, method: str) -> np.ndarray:
    scores = view.mcdm_scores.get(method)
    if scores is None:
        raise ValueError(f"No scores for method '{method}'.")
    order = np.argsort(scores)
    return order


def feasible_indices(view: DecisionView, max_violation: float = 0.0) -> np.ndarray:
    if view.constraints is None:
        return np.arange(view.front.points_F.shape[0])
    violations = np.maximum(view.constraints, 0.0).sum(axis=1)
    return np.nonzero(violations <= max_violation)[0]


def filter_by_objective_ranges(
    view: DecisionView, ranges: Sequence[tuple[float | None, float | None]]
) -> np.ndarray:
    F = view.front.points_F
    if len(ranges) != F.shape[1]:
        raise ValueError("ranges length must equal number of objectives.")
    mask = np.ones(F.shape[0], dtype=bool)
    for idx, (lo, hi) in enumerate(ranges):
        if lo is not None:
            mask &= F[:, idx] >= lo
        if hi is not None:
            mask &= F[:, idx] <= hi
    return np.nonzero(mask)[0]
