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
    return np.asarray(w / w.sum(), dtype=float)


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


def topsis_scores(F: np.ndarray, weights: np.ndarray) -> MCDMResult:
    """
    Calculate TOPSIS scores (closeness coefficient to ideal solution).
    Scores are relative closeness C_i = S_minus / (S_plus + S_minus).
    For MCDMResult compatibility where 'lower score is better' is often assumed,
    we return 1 - C_i as the score (so minimizing score maximizes closeness).
    BUT wait, existing methods return different things.
      - weighted_sum: weighted sum (minimize)
      - tchebycheff: max deviation (minimize)
      - reference_point: distance (minimize)
      - knee_point: negative distance (minimize)

    So consistently, we want to MINIMIZE the returned 'scores'.
    TOPSIS 'C' is in [0, 1], where 1 is best.
    So we will return scores = 1.0 - C.
    """
    F = _validate_front(F)
    w = _validate_weights(weights, F.shape[1])

    # 1. Vector Normalization
    norm = np.linalg.norm(F, axis=0)
    # Handle zero norm (constant objective 0)
    norm = np.where(norm == 0, 1.0, norm)
    F_norm = F / norm

    # 2. Weighted Normalized Decision Matrix
    V = F_norm * w

    # 3. Determine Ideal (Min) and Anti-Ideal (Max) for minimization problems
    # VAMOS assumes minimization convention for F
    A_best = np.min(V, axis=0)
    A_worst = np.max(V, axis=0)

    # 4. Calculate Separation Measures
    S_plus = np.linalg.norm(V - A_best, axis=1)
    S_minus = np.linalg.norm(V - A_worst, axis=1)

    # 5. Calculate Relative Closeness
    denom = S_plus + S_minus
    # Handle division by zero (if point coincides with both ideal and anti-ideal? Impossible unless F is single point)
    denom = np.where(denom == 0, 1e-10, denom)

    C = S_minus / denom

    # We want to maximize C, so we minimize (1 - C)
    scores = 1.0 - C
    best_idx = int(np.argmin(scores))

    return MCDMResult(scores=scores, best_index=best_idx, best_point=F[best_idx].copy())
