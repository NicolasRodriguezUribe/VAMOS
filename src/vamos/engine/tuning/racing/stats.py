from __future__ import annotations

import math
from typing import List, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .state import ConfigState


def build_score_matrix(
    configs: Sequence["ConfigState"],
) -> Tuple[np.ndarray, List[int]]:
    """
    Build a score matrix from the given configs.

    Returns:
        scores: np.ndarray of shape (n_configs, n_blocks)
            Each row corresponds to one config, each column to one "block"
            (instance × seed combination). We assume that scores are aligned
            in time: the t-th element of each config's scores corresponds to
            the same block.
        alive_indices: List[int]
            Indices in the original `configs` sequence corresponding to the
            rows in the returned score matrix (i.e., only alive configs with
            at least one score).
    """
    rows: List[np.ndarray] = []
    alive_indices: List[int] = []

    for idx, state in enumerate(configs):
        if not state.alive or not state.scores:
            continue
        rows.append(np.asarray(state.scores, dtype=float))
        alive_indices.append(idx)

    if not rows:
        return np.empty((0, 0), dtype=float), []

    min_len = min(row.shape[0] for row in rows)
    if min_len == 0:
        return np.empty((0, 0), dtype=float), []

    trimmed_rows = [row[:min_len] for row in rows]
    scores = np.vstack(trimmed_rows)

    return scores, alive_indices


def _z_critical(alpha: float) -> float:
    """
    Return the two-sided critical z-value for the given significance level alpha.

    If scipy is available, use scipy.stats.norm.ppf. Otherwise, support a small
    set of typical alpha values with precomputed constants.
    """
    try:
        from scipy.stats import norm  # type: ignore
    except Exception:
        if abs(alpha - 0.05) < 1e-8:
            return 1.96
        if abs(alpha - 0.10) < 1e-8:
            return 1.64
        if abs(alpha - 0.01) < 1e-8:
            return 2.58
        return 1.96

    return float(norm.ppf(1.0 - alpha / 2.0))


def select_configs_by_paired_test(
    scores: np.ndarray,
    maximize: bool,
    alpha: float,
) -> np.ndarray:
    """
    Given a score matrix of shape (n_configs, n_blocks), perform paired tests
    against the current best configuration and decide which configs to keep.

    Returns:
        keep: boolean array of length n_configs where True means the config is
        NOT significantly worse than the best configuration.
    """
    n_configs, n_blocks = scores.shape
    keep = np.ones(n_configs, dtype=bool)

    if n_configs <= 1 or n_blocks <= 1:
        return keep

    agg_scores = scores.mean(axis=1)
    best_idx = int(np.argmax(agg_scores)) if maximize else int(np.argmin(agg_scores))
    best_scores = scores[best_idx, :]
    z_crit = _z_critical(alpha)

    for i in range(n_configs):
        if i == best_idx:
            continue

        cfg_scores = scores[i, :]
        diffs = best_scores - cfg_scores if maximize else cfg_scores - best_scores

        mean_diff = float(diffs.mean())
        sd_diff = float(diffs.std(ddof=1)) if n_blocks > 1 else 0.0

        if sd_diff <= 1e-12:
            if mean_diff > 0.0:
                keep[i] = False
            continue

        t_stat = mean_diff / (sd_diff / math.sqrt(n_blocks))
        if t_stat > z_crit:
            keep[i] = False

    return keep


__all__ = [
    "build_score_matrix",
    "_z_critical",
    "select_configs_by_paired_test",
]
