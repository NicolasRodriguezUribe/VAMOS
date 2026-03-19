"""
Archive-aware survival helpers for NSGA-II.

This module keeps Pareto rank/front order unchanged and only replaces
split-front truncation when the archive-aware mode is explicitly enabled.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .helpers import compute_front_crowding, fronts_from_ranks

ArchiveHybridNormalization = Literal["minmax_archive_split"]
SUPPORTED_ARCHIVE_HYBRID_NORMALIZATIONS = {"minmax_archive_split"}


def _archive_novelty_fallback_reason(
    split_F: np.ndarray,
    archive_F: np.ndarray | None,
    *,
    k: int,
) -> str | None:
    if archive_F is None:
        return "archive_missing"

    split_arr = np.asarray(split_F, dtype=float)
    archive_arr = np.asarray(archive_F, dtype=float)
    if archive_arr.ndim != 2 or split_arr.ndim != 2 or archive_arr.shape[1] != split_arr.shape[1]:
        return "archive_shape_mismatch"
    if archive_arr.shape[0] == 0:
        return "archive_empty"
    if archive_arr.shape[0] < int(k):
        return "archive_too_small"
    return None


def supports_archive_hybrid_survival(
    state: Any,
    *,
    G_off: np.ndarray | None = None,
) -> tuple[bool, str | None]:
    """Return whether the current NSGA-II state supports hybrid survival."""
    mode = str(getattr(state, "archive_mode", "off") or "off").strip().lower()
    if mode != "hybrid_survival":
        return False, "archive_mode_off"
    if bool(getattr(state, "incremental_mode", False)):
        return False, "incremental_mode"
    if getattr(state, "G", None) is not None or G_off is not None:
        return False, "constraints"
    return True, None


def normalize_scores(
    values: np.ndarray,
    *,
    uniform_value: float = 0.5,
) -> np.ndarray:
    """Normalize a 1-D score vector to [0, 1] with deterministic finite output."""
    scores = np.asarray(values, dtype=float).reshape(-1)
    if scores.size == 0:
        return scores.copy()

    normalized = np.full(scores.shape, float(uniform_value), dtype=float)
    finite_mask = np.isfinite(scores)
    pos_inf_mask = np.isposinf(scores)
    neg_inf_mask = np.isneginf(scores)

    if finite_mask.any():
        finite_values = scores[finite_mask]
        lo = float(np.min(finite_values))
        hi = float(np.max(finite_values))
        span = hi - lo
        if span > 1e-12:
            normalized[finite_mask] = (finite_values - lo) / span
        else:
            normalized[finite_mask] = float(uniform_value)
        normalized[pos_inf_mask] = 1.0
        normalized[neg_inf_mask] = 0.0
    else:
        normalized.fill(float(uniform_value))
        if pos_inf_mask.any() and not np.all(pos_inf_mask):
            normalized[pos_inf_mask] = 1.0
        if neg_inf_mask.any() and not np.all(neg_inf_mask):
            normalized[neg_inf_mask] = 0.0

    return np.nan_to_num(normalized, nan=float(uniform_value), posinf=1.0, neginf=0.0)


def compute_local_split_scores(split_F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute raw and normalized crowding scores for a split front."""
    split_arr = np.asarray(split_F, dtype=float)
    if split_arr.ndim != 2:
        raise ValueError("split_F must be a 2D objective matrix.")
    if split_arr.shape[0] == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    local_raw = compute_front_crowding(split_arr, np.arange(split_arr.shape[0], dtype=int))
    return local_raw, normalize_scores(local_raw)


def _normalize_archive_and_split(
    split_F: np.ndarray,
    archive_F: np.ndarray,
    *,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray]:
    if normalization not in SUPPORTED_ARCHIVE_HYBRID_NORMALIZATIONS:
        raise ValueError("archive_hybrid_normalization must be 'minmax_archive_split'.")

    combined = np.vstack([archive_F, split_F])
    mins = np.min(combined, axis=0)
    maxs = np.max(combined, axis=0)
    denom = np.maximum(maxs - mins, 1e-12)
    split_norm = (split_F - mins) / denom
    archive_norm = (archive_F - mins) / denom
    return split_norm, archive_norm


def compute_archive_novelty_scores(
    split_F: np.ndarray,
    archive_F: np.ndarray | None,
    *,
    k: int,
    normalization: str = "minmax_archive_split",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return raw and normalized archive novelty scores for split-front candidates."""
    split_arr = np.asarray(split_F, dtype=float)
    if split_arr.ndim != 2:
        raise ValueError("split_F must be a 2D objective matrix.")
    if split_arr.shape[0] == 0:
        empty = np.empty(0, dtype=float)
        return empty, empty
    if archive_F is None:
        return None, None

    archive_arr = np.asarray(archive_F, dtype=float)
    if archive_arr.ndim != 2 or archive_arr.shape[1] != split_arr.shape[1]:
        return None, None
    if archive_arr.shape[0] < int(k):
        return None, None

    split_norm, archive_norm = _normalize_archive_and_split(split_arr, archive_arr, normalization=normalization)
    distances = np.linalg.norm(split_norm[:, None, :] - archive_norm[None, :, :], axis=2)
    kth = np.partition(distances, int(k) - 1, axis=1)[:, int(k) - 1]
    novelty_raw = np.asarray(kth, dtype=float)
    return novelty_raw, normalize_scores(novelty_raw)


def combine_hybrid_scores(
    local_scores: np.ndarray,
    archive_scores: np.ndarray | None,
    *,
    alpha: float,
) -> np.ndarray:
    """Combine normalized local and archive scores."""
    local_arr = np.asarray(local_scores, dtype=float)
    if archive_scores is None:
        return local_arr.copy()
    archive_arr = np.asarray(archive_scores, dtype=float)
    return float(alpha) * local_arr + (1.0 - float(alpha)) * archive_arr


def score_hybrid_split_front(
    split_F: np.ndarray,
    archive_F: np.ndarray | None,
    *,
    alpha: float,
    k: int,
    normalization: str = "minmax_archive_split",
) -> dict[str, Any]:
    """Compute local, archive, and combined scores for split-front candidates."""
    local_raw, local_scores = compute_local_split_scores(split_F)
    novelty_fallback_reason = _archive_novelty_fallback_reason(split_F, archive_F, k=int(k))
    archive_raw, archive_scores = compute_archive_novelty_scores(
        split_F,
        archive_F,
        k=int(k),
        normalization=normalization,
    )
    combined_scores = combine_hybrid_scores(local_scores, archive_scores, alpha=float(alpha))
    return {
        "local_raw": local_raw,
        "local_scores": local_scores,
        "archive_raw": archive_raw,
        "archive_scores": archive_scores,
        "combined_scores": combined_scores,
        "used_archive": archive_scores is not None,
        "split_front_mode": "archive" if archive_scores is not None else "local_only",
        "novelty_fallback_reason": None if archive_scores is not None else novelty_fallback_reason,
    }


def rank_split_front_candidates(
    combined_scores: np.ndarray,
    local_scores: np.ndarray,
) -> np.ndarray:
    """Rank split-front candidates by hybrid score, then local score, then stable index."""
    combined_arr = np.asarray(combined_scores, dtype=float).reshape(-1)
    local_arr = np.asarray(local_scores, dtype=float).reshape(-1)
    if combined_arr.shape != local_arr.shape:
        raise ValueError("combined_scores and local_scores must have the same shape.")
    stable_idx = np.arange(combined_arr.shape[0], dtype=int)
    return np.lexsort((stable_idx, -local_arr, -combined_arr))


def select_hybrid_split_front(
    split_F: np.ndarray,
    n_keep: int,
    *,
    archive_F: np.ndarray | None,
    alpha: float,
    k: int,
    normalization: str = "minmax_archive_split",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select survivors from a split front using hybrid local/archive scores."""
    split_arr = np.asarray(split_F, dtype=float)
    if split_arr.ndim != 2:
        raise ValueError("split_F must be a 2D objective matrix.")
    if n_keep <= 0 or split_arr.shape[0] == 0:
        return np.empty(0, dtype=int), score_hybrid_split_front(
            split_arr,
            archive_F,
            alpha=alpha,
            k=k,
            normalization=normalization,
        )

    scores = score_hybrid_split_front(
        split_arr,
        archive_F,
        alpha=alpha,
        k=k,
        normalization=normalization,
    )
    order = rank_split_front_candidates(scores["combined_scores"], scores["local_scores"])
    return order[: min(int(n_keep), split_arr.shape[0])], scores


def archive_aware_nsga2_survival(
    kernel: Any,
    X: np.ndarray,
    F: np.ndarray,
    X_off: np.ndarray,
    F_off: np.ndarray,
    pop_size: int,
    *,
    archive_F: np.ndarray | None,
    alpha: float,
    k: int,
    normalization: str = "minmax_archive_split",
    return_indices: bool = False,
    return_details: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]] | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """NSGA-II survival with archive-aware split-front truncation."""
    X_comb = np.vstack([X, X_off])
    F_comb = np.vstack([F, F_off])
    ranks, _ = kernel.nsga2_ranking(F_comb)
    fronts = fronts_from_ranks(ranks)

    selected: list[int] = []
    split_details: dict[str, Any] = {
        "used_archive": False,
        "split_front_mode": "not_applicable",
        "novelty_fallback_reason": None,
        "split_front_size": 0,
        "selected_split_size": 0,
    }
    for front in fronts:
        if not front:
            continue
        front_arr = np.asarray(front, dtype=int)
        if len(selected) + front_arr.size <= pop_size:
            selected.extend(front_arr.tolist())
            continue

        remaining = pop_size - len(selected)
        local_idx, scores = select_hybrid_split_front(
            F_comb[front_arr],
            remaining,
            archive_F=archive_F,
            alpha=alpha,
            k=k,
            normalization=normalization,
        )
        split_details = {
            "used_archive": bool(scores["used_archive"]),
            "split_front_mode": str(scores["split_front_mode"]),
            "novelty_fallback_reason": scores["novelty_fallback_reason"],
            "split_front_size": int(front_arr.shape[0]),
            "selected_split_size": int(local_idx.shape[0]),
        }
        selected.extend(front_arr[local_idx].tolist())
        break

    selected_idx = np.asarray(selected, dtype=int)
    if return_indices and return_details:
        return X_comb[selected_idx], F_comb[selected_idx], selected_idx, split_details
    if return_indices:
        return X_comb[selected_idx], F_comb[selected_idx], selected_idx
    if return_details:
        return X_comb[selected_idx], F_comb[selected_idx], split_details
    return X_comb[selected_idx], F_comb[selected_idx]


__all__ = [
    "SUPPORTED_ARCHIVE_HYBRID_NORMALIZATIONS",
    "archive_aware_nsga2_survival",
    "combine_hybrid_scores",
    "compute_archive_novelty_scores",
    "compute_local_split_scores",
    "normalize_scores",
    "rank_split_front_candidates",
    "score_hybrid_split_front",
    "select_hybrid_split_front",
    "supports_archive_hybrid_survival",
]
