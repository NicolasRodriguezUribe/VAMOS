from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MCDMResult:
    scores: np.ndarray
    best_index: int
    best_point: np.ndarray


def _validate_front(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] == 0 or F.shape[1] == 0:
        raise ValueError("F must be a 2D array with at least one point and one objective.")
    return F


def _validate_weights(weights: np.ndarray, n_obj: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.shape[0] != n_obj:
        raise ValueError("weights must be 1D with length equal to number of objectives.")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    if np.allclose(w.sum(), 0):
        raise ValueError("weights must not sum to zero.")
    return w / w.sum()


def weighted_sum_scores(F: np.ndarray, weights: np.ndarray) -> MCDMResult:
    F = _validate_front(F)
    w = _validate_weights(weights, F.shape[1])
    scores = F @ w
    best_idx = int(np.argmin(scores))
    return MCDMResult(scores=scores, best_index=best_idx, best_point=F[best_idx].copy())


def tchebycheff_scores(
    F: np.ndarray,
    weights: np.ndarray,
    reference: np.ndarray | None = None,
) -> MCDMResult:
    F = _validate_front(F)
    w = _validate_weights(weights, F.shape[1])
    if reference is None:
        reference = np.min(F, axis=0)
    reference = np.asarray(reference, dtype=float)
    if reference.shape != (F.shape[1],):
        raise ValueError("reference must have shape (n_obj,).")
    diff = np.abs(F - reference)
    scores = np.max(w * diff, axis=1)
    best_idx = int(np.argmin(scores))
    return MCDMResult(scores=scores, best_index=best_idx, best_point=F[best_idx].copy())


def reference_point_scores(F: np.ndarray, reference: np.ndarray) -> MCDMResult:
    F = _validate_front(F)
    ref = np.asarray(reference, dtype=float)
    if ref.shape != (F.shape[1],):
        raise ValueError("reference must have shape (n_obj,).")
    diff = F - ref
    scores = np.linalg.norm(diff, axis=1)
    best_idx = int(np.argmin(scores))
    return MCDMResult(scores=scores, best_index=best_idx, best_point=F[best_idx].copy())


def knee_point_scores(F: np.ndarray) -> MCDMResult:
    F = _validate_front(F)
    if F.shape[1] != 2:
        raise ValueError("knee_point_scores currently supports only 2D fronts.")
    order = np.argsort(F[:, 0])
    sorted_F = F[order]
    p_start = sorted_F[0]
    p_end = sorted_F[-1]
    vec = p_end - p_start
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        scores = np.zeros(F.shape[0], dtype=float)
        return MCDMResult(scores=scores, best_index=0, best_point=F[0].copy())
    vec /= norm
    diffs = sorted_F - p_start
    proj = diffs @ vec
    # proj_point = np.outer(proj, vec) + p_start  # projection onto the line (debug)
    distances = np.linalg.norm(diffs - (proj[:, None] * vec), axis=1)
    best_local_idx = int(np.argmax(distances))
    best_global_idx = int(order[best_local_idx])
    scores = np.empty(F.shape[0], dtype=float)
    scores[order] = -distances  # more negative is better knee
    return MCDMResult(scores=scores, best_index=best_global_idx, best_point=F[best_global_idx].copy())
