from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .state import ConfigState


def build_score_matrix(
    configs: Sequence[ConfigState],
) -> tuple[np.ndarray, list[int]]:
    """
    Build a score matrix from the given configs, trimmed to the minimum history
    length across alive configs.

    Returns:
        scores: np.ndarray of shape (n_configs, n_blocks)
            Each row corresponds to one config, each column to one "block"
            (instance x seed combination). We assume that scores are aligned
            in time: the t-th element of each config's scores corresponds to
            the same block.
        alive_indices: list[int]
            Indices in the original `configs` sequence corresponding to the
            rows in the returned score matrix (i.e., only alive configs with
            at least one score).
    """
    rows: list[np.ndarray] = []
    alive_indices: list[int] = []

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
        from scipy.stats import norm  # type: ignore[import-untyped]
    except Exception:
        if abs(alpha - 0.05) < 1e-8:
            return 1.96
        if abs(alpha - 0.10) < 1e-8:
            return 1.64
        if abs(alpha - 0.01) < 1e-8:
            return 2.58
        return 1.96

    return float(norm.ppf(1.0 - alpha / 2.0))


def _t_critical(alpha: float, df: int) -> float:
    """
    Two-sided critical t-value for the given alpha and degrees of freedom.

    Falls back to z critical when scipy is unavailable.
    """
    try:
        from scipy.stats import t
    except Exception:
        return _z_critical(alpha)

    return float(t.ppf(1.0 - alpha / 2.0, df))


def _get_p_value(t_stat: float, df: int) -> float:
    """
    Two-sided p-value for a given t-statistic and degrees of freedom.
    """
    try:
        from scipy.stats import t

        # Survival function (1 - cdf) for the absolute t-stat * 2 for two-sided
        return float(t.sf(abs(t_stat), df) * 2)
    except Exception:
        # Fallback to Normal distribution if scipy is missing
        from math import erf, sqrt

        # z-test p-value
        # 1 - erf(z / sqrt(2)) is 2 * (1 - Phi(z)) for one-sided?
        # standard normal cdf Phi(z) = 0.5 * (1 + erf(z/sqrt(2)))
        # p = 2 * (1 - Phi(|z|)) = 2 * (1 - 0.5 * (1 + erf(|z|/sqrt(2))))
        #   = 2 * 0.5 * (1 - erf(|z|/sqrt(2))) = 1 - erf(|z|/sqrt(2))
        return 1.0 - erf(abs(t_stat) / sqrt(2.0))


def select_configs_by_paired_test(
    scores: np.ndarray,
    maximize: bool,
    alpha: float,
    *,
    aggregator: Callable[[list[float]], float] | None = None,
) -> np.ndarray:
    """
    Perform pairwise t-tests against the best configuration with Holm-Bonferroni correction.

    Returns:
        keep: boolean array where True means the config is kept (not eliminated).
    """
    n_configs, n_blocks = scores.shape
    keep = np.ones(n_configs, dtype=bool)

    if n_configs <= 1 or n_blocks <= 1:
        return keep

    if aggregator is None:
        agg_scores = scores.mean(axis=1)
    else:
        agg_scores = np.asarray([float(aggregator(row.tolist())) for row in scores], dtype=float)

    best_idx = int(np.argmax(agg_scores)) if maximize else int(np.argmin(agg_scores))
    best_scores = scores[best_idx, :]

    # Store tuples of (p_value, config_index)
    comparisons = []

    for i in range(n_configs):
        if i == best_idx:
            continue

        cfg_scores = scores[i, :]
        diffs = best_scores - cfg_scores if maximize else cfg_scores - best_scores

        mean_diff = float(diffs.mean())

        # If mean difference is negative, the candidate is actually *better* (or equal)
        # than our chosen 'best' in this sample (could happen if aggregator != mean).
        # In that case, we definitely keep it.
        # Even if aggregator == mean, float precision might cause slight mismatches,
        # but generally mean_diff >= 0 if i != best_idx for mean aggregator.
        # If mean_diff <= 0, we treat it as "not significantly worse".
        if mean_diff <= 0:
            continue

        sd_diff = float(diffs.std(ddof=1)) if n_blocks > 1 else 0.0

        if sd_diff <= 1e-12:
            # Deterministically worse
            # p-value ~ 0
            comparisons.append((0.0, i))
            continue

        t_stat = mean_diff / (sd_diff / math.sqrt(n_blocks))
        p_val = _get_p_value(t_stat, df=n_blocks - 1)
        comparisons.append((p_val, i))

    if not comparisons:
        return keep

    # Holm-Bonferroni Step-Down Procedure
    # 1. Sort p-values from smallest to largest
    comparisons.sort(key=lambda x: x[0])

    m = len(comparisons)  # Total number of hypotheses (candidates vs best)

    for k, (p_val, idx) in enumerate(comparisons):
        # Rank k is 0-based here, so it corresponds to k=1..m in literature
        # Adjusted alpha = alpha / (m - k)
        adj_alpha = alpha / (m - k)

        if p_val < adj_alpha:
            # Reject H0 -> Significantly worse -> Eliminate
            keep[idx] = False
        else:
            # Fail to reject H0 -> Stop eliminating
            # Since p-values are sorted, all subsequent p-values are larger
            # and denominators (m-k) are smaller, so they will also pass.
            break

    return keep


__all__ = [
    "build_score_matrix",
    "select_configs_by_paired_test",
]
