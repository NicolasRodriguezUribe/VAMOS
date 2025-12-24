from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .stats import build_score_matrix, select_configs_by_paired_test
from .state import ConfigState, EliteEntry


def compute_aggregated_scores(configs: List[ConfigState], task) -> Tuple[List[int], np.ndarray]:
    """
    Compute aggregated scores for all alive configs that have at least one score.
    """
    alive_indices: List[int] = []
    agg_values: List[float] = []
    for idx, state in enumerate(configs):
        if not state.alive or not state.scores:
            continue
        alive_indices.append(idx)
        agg_values.append(float(task.aggregator(state.scores)))
    if not agg_values:
        return [], np.array([], dtype=float)
    return alive_indices, np.asarray(agg_values, dtype=float)


def rank_based_elimination(
    configs: List[ConfigState],
    alive_indices: List[int],
    agg_scores: np.ndarray,
    *,
    task,
    scenario,
) -> bool:
    n_alive = len(alive_indices)
    if n_alive <= 1:
        return False

    order = np.argsort(-agg_scores if task.maximize else agg_scores)

    target_keep = max(
        scenario.min_survivors,
        int(math.ceil(n_alive * (1.0 - scenario.elimination_fraction))),
    )
    target_keep = max(1, min(target_keep, n_alive))

    keep_rows = set(int(idx) for idx in order[:target_keep])
    eliminated_any = False
    for row_idx, cfg_idx in enumerate(alive_indices):
        if row_idx not in keep_rows:
            if configs[cfg_idx].alive:
                configs[cfg_idx].alive = False
                eliminated_any = True
    return eliminated_any


def force_keep_top_k(
    configs: List[ConfigState],
    alive_indices: List[int],
    agg_scores: np.ndarray,
    *,
    task,
    k: int,
) -> bool:
    """Ensure that at least k configs remain alive by keeping the k best."""
    n_alive = len(alive_indices)
    if n_alive <= k:
        return False

    order = np.argsort(-agg_scores if task.maximize else agg_scores)
    keep_rows = set(int(idx) for idx in order[:k])

    eliminated_any = False
    for row_idx, cfg_idx in enumerate(alive_indices):
        new_alive = row_idx in keep_rows
        if configs[cfg_idx].alive != new_alive:
            configs[cfg_idx].alive = new_alive
            eliminated_any = True
    return eliminated_any


def eliminate_configs(configs: List[ConfigState], *, task, scenario) -> bool:
    """
    Eliminate configurations based on the current scores.
    """
    alive_indices, agg_scores = compute_aggregated_scores(configs, task)
    if agg_scores.size == 0 or len(alive_indices) <= 1:
        return False

    lengths = [len(configs[idx].scores) for idx in alive_indices]
    min_len = min(lengths)
    max_len = max(lengths)

    use_stat_tests = (
        scenario.use_statistical_tests
        and min_len >= scenario.min_blocks_before_elimination
        and min_len > 1
    )

    if not use_stat_tests:
        return rank_based_elimination(
            configs,
            alive_indices,
            agg_scores,
            task=task,
            scenario=scenario,
        )

    if min_len != max_len:
        return rank_based_elimination(
            configs,
            alive_indices,
            agg_scores,
            task=task,
            scenario=scenario,
        )

    scores, aligned_indices = build_score_matrix(configs)

    # --- Friedman Test Pre-Check ---
    # Only if we have enough samples to be meaningful (e.g. at least 3 blocks, at least 3 configs)
    if len(aligned_indices) >= 3 and scores.shape[1] >= 3:
        try:
            from scipy.stats import friedmanchisquare
            # friedmanchisquare takes arguments as *samples
            # each sample is an array of measurements for one subject (here config)
            # We must pass rows of the score matrix
            stat, p_friedman = friedmanchisquare(*[scores[i, :] for i in range(scores.shape[0])])
            
            # If p-value is high, we cannot reject the null hypothesis that all configs are equivalent.
            # Thus, we should NOT eliminate anyone yet.
            if p_friedman > scenario.alpha:
                return False
        except ImportError:
            pass  # no scipy, skip pre-check
        except ValueError:
            pass  # can happen if all numbers are identical
    # -------------------------------

    keep_mask = select_configs_by_paired_test(
        scores=scores,
        maximize=task.maximize,
        alpha=scenario.alpha,
        aggregator=task.aggregator,
    )

    num_keep = int(keep_mask.sum())
    if num_keep <= 0:
        best_idx = int(np.argmax(agg_scores)) if task.maximize else int(np.argmin(agg_scores))
        keep_mask[best_idx] = True
        num_keep = 1

    if num_keep < scenario.min_survivors:
        return force_keep_top_k(
            configs,
            aligned_indices,
            agg_scores,
            task=task,
            k=scenario.min_survivors,
        )

    eliminated_any = False
    for row_idx, cfg_idx in enumerate(aligned_indices):
        if not keep_mask[row_idx]:
            if configs[cfg_idx].alive:
                configs[cfg_idx].alive = False
                eliminated_any = True
    return eliminated_any


def update_elite_archive(
    configs: List[ConfigState],
    *,
    task,
    scenario,
    elite_archive: List[EliteEntry],
) -> List[EliteEntry]:
    """
    Update the elite archive based on the current alive configurations.
    """
    if not scenario.use_elitist_restarts:
        return elite_archive

    indices, agg_scores = compute_aggregated_scores(configs, task)
    if not indices:
        return elite_archive

    n_alive = len(indices)
    if n_alive == 0:
        return elite_archive

    k = max(1, int(math.ceil(scenario.elite_fraction * n_alive)))

    if task.maximize:
        order = np.argsort(-agg_scores)
    else:
        order = np.argsort(agg_scores)

    elite_entries: List[EliteEntry] = []
    for rank in range(min(k, n_alive)):
        row_idx = int(order[rank])
        cfg_idx = indices[row_idx]
        state = configs[cfg_idx]
        score = float(agg_scores[row_idx])
        elite_entries.append(EliteEntry(config=dict(state.config), score=score))

    all_elites = elite_archive + elite_entries
    if not all_elites:
        return []

    if task.maximize:
        all_elites.sort(key=lambda e: e.score, reverse=True)
    else:
        all_elites.sort(key=lambda e: e.score)

    return all_elites[: scenario.max_elite_archive_size]


__all__ = [
    "compute_aggregated_scores",
    "eliminate_configs",
    "rank_based_elimination",
    "force_keep_top_k",
    "update_elite_archive",
]
