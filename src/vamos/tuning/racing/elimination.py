from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from ..stats import build_score_matrix, select_configs_by_paired_test
from .state import ConfigState, EliteEntry


def aggregate_rows(task, scores: np.ndarray) -> np.ndarray:
    """Apply task aggregator row-wise to a score matrix."""
    return np.asarray([float(task.aggregator(row.tolist())) for row in scores], dtype=float)


def compute_aggregated_scores(configs: List[ConfigState], task) -> Tuple[List[int], np.ndarray]:
    """
    Compute aggregated scores for all alive configs that have at least one score.
    """
    scores, alive_indices = build_score_matrix(configs)
    if scores.size == 0 or len(alive_indices) == 0:
        return [], np.array([], dtype=float)

    agg_values = aggregate_rows(task, scores)
    return alive_indices, np.asarray(agg_values, dtype=float)


def rank_based_elimination(
    configs: List[ConfigState],
    scores: np.ndarray,
    alive_indices: List[int],
    *,
    task,
    scenario,
) -> bool:
    n_alive = len(alive_indices)
    if n_alive <= 1:
        return False

    agg_scores = aggregate_rows(task, scores)
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
    scores: np.ndarray,
    alive_indices: List[int],
    *,
    task,
    k: int,
) -> bool:
    """Ensure that at least k configs remain alive by keeping the k best."""
    n_alive = len(alive_indices)
    if n_alive <= k:
        return False

    agg_scores = aggregate_rows(task, scores)
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
    scores, alive_indices = build_score_matrix(configs)
    if scores.size == 0 or len(alive_indices) <= 1:
        return False

    _, n_blocks = scores.shape

    if (
        not scenario.use_statistical_tests
        or n_blocks < scenario.min_blocks_before_elimination
    ):
        return rank_based_elimination(
            configs,
            scores,
            alive_indices,
            task=task,
            scenario=scenario,
        )

    keep_mask = select_configs_by_paired_test(
        scores=scores,
        maximize=task.maximize,
        alpha=scenario.alpha,
    )

    num_keep = int(keep_mask.sum())
    if num_keep <= 0:
        agg_scores = aggregate_rows(task, scores)
        best_idx = int(np.argmax(agg_scores)) if task.maximize else int(np.argmin(agg_scores))
        keep_mask[best_idx] = True
        num_keep = 1

    if num_keep < scenario.min_survivors:
        return force_keep_top_k(
            configs,
            scores,
            alive_indices,
            task=task,
            k=scenario.min_survivors,
        )

    eliminated_any = False
    for row_idx, cfg_idx in enumerate(alive_indices):
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
    "aggregate_rows",
    "compute_aggregated_scores",
    "eliminate_configs",
    "rank_based_elimination",
    "force_keep_top_k",
    "update_elite_archive",
]
