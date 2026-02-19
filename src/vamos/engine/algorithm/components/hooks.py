"""
Hook integration points for live visualization and genealogy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


if TYPE_CHECKING:
    from vamos.engine.algorithm.components.state import AlgorithmState
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.hooks.genealogy import GenealogyTracker
    from vamos.hooks.live_viz import LiveVisualization


def get_live_viz(
    live_viz: LiveVisualization | None,
) -> LiveVisualization:
    """
    Get live visualization callback, defaulting to no-op.

    Parameters
    ----------
    live_viz : LiveVisualization | None
        User-provided callback or None.

    Returns
    -------
    LiveVisualization
        The callback or a no-op implementation.
    """
    from vamos.hooks.live_viz import NoOpLiveVisualization

    return live_viz or NoOpLiveVisualization()


def notify_generation(
    live_cb: LiveVisualization,
    kernel: KernelBackend,
    generation: int,
    F: np.ndarray,
    stats: dict[str, Any] | None = None,
) -> bool:
    """
    Notify live visualization of generation progress.

    Parameters
    ----------
    live_cb : LiveVisualization
        Live visualization callback.
    kernel : KernelBackend
        Kernel for computing non-dominated front.
    generation : int
        Current generation number.
    F : np.ndarray
        Current population objectives.
    stats : dict[str, Any] | None
        Optional metrics payload (e.g., evaluation counts).

    Returns
    -------
    bool
        True if the callback requests stopping.
    """
    try:
        ranks, _ = kernel.nsga2_ranking(F)
        nd_mask = ranks == ranks.min(initial=0)
        live_cb.on_generation(generation, F=F[nd_mask], stats=stats)
    except (ValueError, IndexError) as exc:
        _logger().debug("Failed to compute non-dominated front for viz: %s", exc)
        live_cb.on_generation(generation, F=F, stats=stats)
    return live_should_stop(live_cb)


def live_should_stop(live_cb: LiveVisualization) -> bool:
    should_stop = getattr(live_cb, "should_stop", None)
    if not callable(should_stop):
        return False
    try:
        return bool(should_stop())
    except Exception:
        return False


def setup_genealogy(
    pop_size: int,
    F: np.ndarray,
    track_genealogy: bool,
    algorithm_name: str = "algorithm",
) -> tuple[GenealogyTracker | None, np.ndarray | None]:
    """
    Initialize genealogy tracking if enabled.

    Parameters
    ----------
    pop_size : int
        Population size.
    F : np.ndarray
        Initial fitness values.
    track_genealogy : bool
        Whether to enable genealogy tracking.
    algorithm_name : str
        Algorithm name for tracking records.

    Returns
    -------
    tuple[GenealogyTracker | None, np.ndarray | None]
        (genealogy_tracker, individual_ids)
    """
    if not track_genealogy:
        return None, None

    from vamos.hooks.genealogy import DefaultGenealogyTracker

    genealogy_tracker = DefaultGenealogyTracker()
    ids = np.arange(pop_size, dtype=int)
    for i in range(pop_size):
        genealogy_tracker.new_individual(
            generation=0,
            parents=[],
            operator_name=None,
            algorithm_name=algorithm_name,
            fitness=F[i] if F is not None and i < F.shape[0] else None,
        )
    return genealogy_tracker, ids


def track_offspring_genealogy(
    state: AlgorithmState,
    parent_idx: np.ndarray,
    offspring_count: int,
    operator_name: str = "variation",
    algorithm_name: str = "algorithm",
) -> None:
    """
    Record genealogy for new offspring.

    Parameters
    ----------
    state : AlgorithmState
        Algorithm state with genealogy fields.
    parent_idx : np.ndarray
        Indices of parents used to create offspring.
    offspring_count : int
        Number of offspring created.
    operator_name : str
        Name of the variation operator used.
    algorithm_name : str
        Algorithm name for tracking.
    """
    if not state.track_genealogy or state.genealogy_tracker is None:
        return

    parents_per_offspring = max(1, len(parent_idx) // offspring_count)
    pending_ids = []

    for i in range(offspring_count):
        start = i * parents_per_offspring
        end = min(start + parents_per_offspring, len(parent_idx))
        parent_ids = []
        if state.ids is not None:
            parent_ids = [int(state.ids[p]) for p in parent_idx[start:end] if p < len(state.ids)]

        new_id = state.genealogy_tracker.new_individual(
            generation=state.generation + 1,
            parents=parent_ids,
            operator_name=operator_name,
            algorithm_name=algorithm_name,
            fitness=None,
        )
        pending_ids.append(new_id)

    state.pending_offspring_ids = np.array(pending_ids, dtype=int)


def finalize_genealogy(
    result: dict[str, Any],
    state: AlgorithmState,
    kernel: KernelBackend,
) -> None:
    """
    Finalize genealogy tracking and add to result.

    Parameters
    ----------
    result : dict
        Result dictionary to update.
    state : AlgorithmState
        Algorithm state with genealogy tracker.
    kernel : KernelBackend
        Kernel backend for ranking computation.
    """
    if not state.track_genealogy or state.genealogy_tracker is None:
        return

    try:
        from vamos.engine.algorithm.nsgaii.helpers import (
            generation_contributions,
            operator_success_stats,
        )

        ranks, _ = kernel.nsga2_ranking(state.F)
        nd_mask = ranks == ranks.min(initial=0)
        final_front_ids = state.ids[nd_mask] if state.ids is not None else []
        state.genealogy_tracker.mark_final_front(list(final_front_ids))

        result["genealogy"] = {
            "operator_stats": operator_success_stats(state.genealogy_tracker, list(final_front_ids)),
            "generation_contributions": generation_contributions(state.genealogy_tracker, list(final_front_ids)),
        }
    except (ValueError, IndexError, AttributeError) as exc:
        logging.getLogger(__name__).warning("Failed to compute genealogy stats: %s", exc)


def match_ids(
    new_X: np.ndarray,
    combined_X: np.ndarray,
    combined_ids: np.ndarray,
) -> np.ndarray:
    """
    Match IDs after survival selection by finding matching rows.

    Parameters
    ----------
    new_X : np.ndarray
        New population after survival selection.
    combined_X : np.ndarray
        Combined parent+offspring population before selection.
    combined_ids : np.ndarray
        IDs corresponding to combined_X.

    Returns
    -------
    np.ndarray
        IDs for the new population.
    """
    new_ids = np.zeros(new_X.shape[0], dtype=int)
    for i, row in enumerate(new_X):
        for j, comb_row in enumerate(combined_X):
            if np.allclose(row, comb_row, rtol=1e-12, atol=1e-12):
                new_ids[i] = combined_ids[j]
                break
    return new_ids


__all__ = [
    "get_live_viz",
    "notify_generation",
    "live_should_stop",
    "setup_genealogy",
    "track_offspring_genealogy",
    "finalize_genealogy",
    "match_ids",
]
