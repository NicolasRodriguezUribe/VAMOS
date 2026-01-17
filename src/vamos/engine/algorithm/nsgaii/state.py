# algorithm/nsgaii/state.py
"""
State container and result building for NSGA-II.

This module provides the NSGAIIState dataclass and result-building functions,
keeping the main algorithm file focused on the evolutionary loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

import numpy as np

from vamos.engine.algorithm.components.archive import CrowdingDistanceArchive, HypervolumeArchive, UnboundedArchive
from vamos.engine.algorithm.components.variation import VariationPipeline
from vamos.engine.algorithm.components.termination import HVTracker
from vamos.hooks.genealogy import GenealogyTracker
from vamos.operators.impl.real import VariationWorkspace
from vamos.adaptation.aos.controller import AOSController


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


@dataclass
class NSGAIIState:
    """Mutable state container for NSGA-II algorithm."""

    # Population
    X: np.ndarray
    F: np.ndarray
    G: np.ndarray | None
    rng: np.random.Generator

    # Variation
    variation: VariationPipeline
    operator_pool: list[VariationPipeline]
    variation_workspace: VariationWorkspace

    # Selection
    sel_method: str = "tournament"
    pressure: int = 2

    # Sizes
    pop_size: int = 100
    offspring_size: int = 100

    # Constraints
    constraint_mode: str = "feasibility"

    # Archive
    archive_size: int | None = None
    archive_X: np.ndarray | None = None
    archive_F: np.ndarray | None = None
    archive_manager: CrowdingDistanceArchive | UnboundedArchive | None = None
    archive_via_kernel: bool = False
    result_archive: HypervolumeArchive | CrowdingDistanceArchive | None = None
    result_mode: str = "population"

    # Termination
    hv_tracker: HVTracker | None = None

    # Genealogy
    track_genealogy: bool = False
    genealogy_tracker: GenealogyTracker | None = None
    ids: np.ndarray | None = None

    # Generation tracking
    generation: int = 0

    # Adaptive operator selection (AOS)
    aos_controller: AOSController | None = None
    aos_trace_rows: list[dict[str, Any]] = field(default_factory=list)
    aos_last_op_id: str | None = None
    aos_last_op_name: str | None = None
    aos_last_batch_size: int | None = None
    aos_step: int | None = None

    # Pending offspring (from ask)
    pending_offspring: np.ndarray | None = None
    pending_offspring_ids: np.ndarray | None = None

    # HV points function (computed lazily)
    _hv_points_fn: Callable[[], np.ndarray] | None = field(default=None, repr=False)

    def hv_points_fn(self) -> np.ndarray:
        """Get points for hypervolume computation (archive if available, else population)."""
        if self.archive_F is not None and self.archive_F.size > 0:
            return self.archive_F
        return self.F


def build_result(
    state: NSGAIIState,
    n_eval: int,
    hv_reached: bool,
    kernel: Any = None,
) -> dict[str, Any]:
    """Build the result dictionary from algorithm state.

    Parameters
    ----------
    state : NSGAIIState
        Current algorithm state.
    n_eval : int
        Total number of evaluations.
    hv_reached : bool
        Whether HV threshold was reached.
    kernel : KernelBackend, optional
        Kernel for computing non-dominated ranking. If provided, result
        will contain only non-dominated solutions.

    Returns
    -------
    dict[str, Any]
        Result dictionary with X, F, evaluations, population, and optional archive.
        X and F contain only non-dominated solutions when kernel is provided.
        Full population is always available in 'population' key.
    """
    # Filter to non-dominated solutions only (if requested)
    mode = getattr(state, "result_mode", "population")
    should_filter = kernel is not None and mode is not None and mode != "population"

    if should_filter:
        try:
            ranks, _ = kernel.nsga2_ranking(state.F)
            nd_mask = ranks == ranks.min(initial=0)
            result_X = state.X[nd_mask]
            result_F = state.F[nd_mask]
            result_G = state.G[nd_mask] if state.G is not None else None
        except (ValueError, IndexError) as exc:
            _logger().warning("Failed to filter non-dominated solutions: %s", exc)
            result_X, result_F, result_G = state.X, state.F, state.G
    else:
        result_X, result_F, result_G = state.X, state.F, state.G

    result: dict[str, Any] = {
        "X": result_X,
        "F": result_F,
        "evaluations": n_eval,
        "hv_reached": hv_reached,
        "population": {"X": state.X, "F": state.F},  # Full population always available
    }
    if result_G is not None:
        result["G"] = result_G

    # Add archive contents
    if state.result_archive is not None:
        arch_X, arch_F = state.result_archive.contents()
        result["archive"] = {"X": arch_X, "F": arch_F}
    elif state.archive_size:
        archive_contents = get_archive_contents(state)
        if archive_contents is not None:
            result["archive"] = archive_contents

    if state.aos_controller is not None:
        summary_rows = []
        for row in state.aos_controller.summary_rows():
            summary_rows.append(
                {
                    "op_id": row.op_id,
                    "op_name": row.op_name,
                    "pulls": row.pulls,
                    "mean_reward": row.mean_reward,
                    "total_reward": row.total_reward,
                    "usage_fraction": row.usage_fraction,
                }
            )
        result["aos"] = {
            "trace_rows": list(state.aos_trace_rows),
            "summary": summary_rows,
        }

    return result


def get_archive_contents(state: NSGAIIState) -> dict[str, Any] | None:
    """Extract archive contents from state.

    Parameters
    ----------
    state : NSGAIIState
        Current algorithm state.

    Returns
    -------
    dict[str, Any] | None
        Archive contents with X and F, or None if no archive.
    """
    if state.archive_manager is not None:
        final_X, final_F = state.archive_manager.contents()
        return {"X": final_X, "F": final_F}
    elif state.archive_via_kernel and state.archive_X is not None:
        return {"X": state.archive_X, "F": state.archive_F}
    return None


def finalize_genealogy(
    result: dict[str, Any],
    state: NSGAIIState,
    kernel: Any,
) -> None:
    """Add genealogy stats to result if tracking is enabled.

    Parameters
    ----------
    result : dict[str, Any]
        Result dictionary to update.
    state : NSGAIIState
        Current algorithm state.
    kernel : KernelBackend
        Kernel for ranking computation.
    """
    from .helpers import operator_success_stats, generation_contributions

    if not state.track_genealogy or state.genealogy_tracker is None:
        return

    try:
        ranks, _ = kernel.nsga2_ranking(state.F)
        nd_mask = ranks == ranks.min(initial=0)
        final_front_ids = state.ids[nd_mask] if state.ids is not None else []
        state.genealogy_tracker.mark_final_front(list(final_front_ids))
        result["genealogy"] = {
            "operator_stats": operator_success_stats(state.genealogy_tracker, list(final_front_ids)),
            "generation_contributions": generation_contributions(state.genealogy_tracker, list(final_front_ids)),
        }
    except (ValueError, IndexError, AttributeError) as exc:
        _logger().warning("Failed to compute genealogy stats: %s", exc)


def compute_selection_metrics(
    kernel: Any,
    F: np.ndarray,
    G: np.ndarray | None,
    constraint_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute selection metrics (ranks and crowding) with constraint handling.

    Parameters
    ----------
    kernel : KernelBackend
        Kernel for ranking computation.
    F : np.ndarray
        Objective values.
    G : np.ndarray | None
        Constraint values (None if unconstrained).
    constraint_mode : str
        Constraint handling mode.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (ranks, crowding) arrays.
    """
    from vamos.foundation.constraints.utils import compute_violation, is_feasible

    ranks, crowding = kernel.nsga2_ranking(F)
    if G is not None and constraint_mode != "none":
        cv = compute_violation(G)
        feas = is_feasible(G)
        if feas.any():
            feas_idx = np.nonzero(feas)[0]
            feas_ranks, feas_crowd = kernel.nsga2_ranking(F[feas_idx])
            ranks = np.full(F.shape[0], feas_ranks.max(initial=0) + 1, dtype=int)
            crowding = np.zeros(F.shape[0], dtype=float)
            ranks[feas_idx] = feas_ranks
            crowding[feas_idx] = feas_crowd
            crowding[~feas] = -cv[~feas]
        else:
            ranks = np.zeros(F.shape[0], dtype=int)
            crowding = -cv
    return ranks, crowding


def track_offspring_genealogy(
    state: NSGAIIState,
    parent_idx: np.ndarray,
    n_offspring: int,
) -> None:
    """Track genealogy for generated offspring.

    Parameters
    ----------
    state : NSGAIIState
        Current algorithm state (modified in place).
    parent_idx : np.ndarray
        Indices of parents used.
    n_offspring : int
        Number of offspring generated.
    """
    if not state.track_genealogy or state.genealogy_tracker is None:
        return

    operator_name = f"{state.variation.cross_method}+{state.variation.mut_method}"
    group_size = state.variation.parents_per_group
    children_per_group = state.variation.children_per_group
    parent_groups = parent_idx.reshape(-1, group_size)
    child_ids = []
    gen = state.generation + 1

    for parents in parent_groups:
        parent_ids = state.ids[parents] if state.ids is not None else []
        for _ in range(children_per_group):
            child_ids.append(
                state.genealogy_tracker.new_individual(
                    generation=gen,
                    parents=list(parent_ids),
                    operator_name=operator_name,
                    algorithm_name="nsgaii",
                )
            )
    state.pending_offspring_ids = np.asarray(child_ids[:n_offspring], dtype=int)


def update_archives(
    state: NSGAIIState,
    kernel: Any,
    *,
    X: np.ndarray | None = None,
    F: np.ndarray | None = None,
) -> None:
    """Update result archive and external archive.

    Parameters
    ----------
    state : NSGAIIState
        Current algorithm state (modified in place).
    kernel : KernelBackend
        Kernel backend (may have update_archive method).
    X : np.ndarray | None
        Candidate decision variables to insert (defaults to state.X).
    F : np.ndarray | None
        Candidate objectives to insert (defaults to state.F).
    """
    X_use = state.X if X is None else X
    F_use = state.F if F is None else F
    if state.result_archive is not None:
        state.result_archive.update(X_use, F_use)

    if state.archive_size:
        if state.archive_via_kernel:
            state.archive_X, state.archive_F = kernel.update_archive(
                state.archive_X,
                state.archive_F,
                X_use,
                F_use,
                state.archive_size,
            )
        elif state.archive_manager is not None:
            state.archive_X, state.archive_F = state.archive_manager.update(X_use, F_use)
