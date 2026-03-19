# algorithm/nsgaii/state.py
"""
State container and result building for NSGA-II.

This module provides the NSGAIIState dataclass and result-building functions,
keeping the main algorithm file focused on the evolutionary loop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vamos.engine.adaptation.aos.controller import AOSController
from vamos.engine.algorithm.components.archive import (
    CrowdingDistanceArchive,
    HypervolumeArchive,
    MaxMinArchive,
    ReferenceDirectionsArchive,
    SPEA2Archive,
    UnboundedArchive,
)
from vamos.engine.algorithm.components.results import (
    get_external_archive_payload,
    wants_population_result,
)
from vamos.engine.algorithm.components.subset_selection import select_top_k_crowding
from vamos.engine.algorithm.components.termination import HVTracker
from vamos.engine.algorithm.components.variation import VariationPipeline
from vamos.engine.hooks.genealogy import GenealogyTracker
from vamos.engine.operators.impl.real import VariationWorkspace

ARCHIVE_SUBSET_SELECTOR = "crowding"
ARCHIVE_POPULATION_RESULT_MODES = {"passive", "hybrid_survival"}


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
    replacement_size: int = 1
    incremental_mode: bool = False

    # Constraints
    constraint_mode: str = "feasibility"

    # Archive
    archive_size: int | None = None
    archive_X: np.ndarray | None = None
    archive_F: np.ndarray | None = None
    archive_G: np.ndarray | None = None
    archive_manager: (
        CrowdingDistanceArchive | HypervolumeArchive | MaxMinArchive | ReferenceDirectionsArchive | SPEA2Archive | UnboundedArchive | None
    ) = None
    result_archive: HypervolumeArchive | CrowdingDistanceArchive | MaxMinArchive | ReferenceDirectionsArchive | SPEA2Archive | None = None
    result_mode: str = "non_dominated"
    archive_mode: str = "off"
    archive_subset_size: int | None = None
    archive_hybrid_alpha: float = 0.5
    archive_hybrid_k: int = 3
    archive_hybrid_normalization: str = "minmax_archive_split"
    archive_hybrid_last_status: str = "inactive"
    archive_hybrid_fallback_reason: str | None = None
    archive_hybrid_last_split_mode: str = "inactive"
    archive_hybrid_last_split_reason: str | None = None
    archive_hybrid_generations: int = 0
    archive_hybrid_archive_reference_generations: int = 0
    archive_hybrid_local_only_generations: int = 0
    archive_hybrid_no_split_generations: int = 0

    # Termination
    hv_tracker: HVTracker | None = None

    # Genealogy
    track_genealogy: bool = False
    genealogy_tracker: GenealogyTracker | None = None
    ids: np.ndarray | None = None

    # Generation tracking
    generation: int = 0
    step: int = 0
    replacements: int = 0

    # Cached selection metrics (incremental replacement)
    fronts: list[list[int]] | None = None
    ranks: np.ndarray | None = None
    crowding: np.ndarray | None = None
    incremental_enabled: bool = False

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

    # Optional extension hooks
    immigration_manager: Any | None = None
    parent_selection_filter: Any | None = None
    non_breeding_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    live_callback_mode: str = "nd_only"
    generation_callback: Any | None = None
    generation_callback_copy: bool = True

    # HV points function (computed lazily)
    _hv_points_fn: Callable[[], np.ndarray] | None = field(default=None, repr=False)

    def hv_points_fn(self) -> np.ndarray:
        """Get points for hypervolume computation (archive if available, else population)."""
        if self.archive_mode not in ARCHIVE_POPULATION_RESULT_MODES and self.archive_F is not None and self.archive_F.size > 0:
            return self.archive_F
        return self.F


def _archive_subset_target_size(state: NSGAIIState, archive_size: int) -> int:
    target = state.archive_subset_size
    if target is None:
        target = int(state.pop_size)
    target = int(target)
    if target <= 0:
        raise ValueError("archive_subset_size must be a positive integer.")
    return min(target, archive_size)


def _build_archive_subset_payload(
    state: NSGAIIState,
    archive_payload: dict[str, Any],
) -> dict[str, Any]:
    archive_F = np.asarray(archive_payload["F"], dtype=float)
    archive_X_raw = archive_payload.get("X")
    archive_G_raw = archive_payload.get("G")
    archive_X = None if archive_X_raw is None else np.asarray(archive_X_raw)
    archive_G = None if archive_G_raw is None else np.asarray(archive_G_raw, dtype=float)
    archive_size = int(archive_F.shape[0])
    target_size = _archive_subset_target_size(state, archive_size) if archive_size > 0 else 0

    if archive_size == 0:
        empty_indices = np.empty(0, dtype=int)
        subset_payload: dict[str, Any] = {
            "F": archive_F[:0],
            "indices": empty_indices,
            "selector": ARCHIVE_SUBSET_SELECTOR,
            "size": 0,
            "target_size": target_size,
        }
        if archive_X is not None:
            subset_payload["X"] = archive_X[:0]
        if archive_G is not None:
            subset_payload["G"] = archive_G[:0]
        return subset_payload

    subset_idx = select_top_k_crowding(archive_F, target_size)
    subset_payload = {
        "F": archive_F[subset_idx],
        "indices": np.asarray(subset_idx, dtype=int),
        "selector": ARCHIVE_SUBSET_SELECTOR,
        "size": int(subset_idx.shape[0]),
        "target_size": target_size,
    }
    if archive_X is not None:
        subset_payload["X"] = archive_X[subset_idx]
    if archive_G is not None:
        subset_payload["G"] = archive_G[subset_idx]
    return subset_payload


def _build_archive_export_payload(
    state: NSGAIIState,
    archive_payload: dict[str, Any],
) -> dict[str, Any]:
    archive_export = dict(archive_payload)
    archive_size = int(np.asarray(archive_export["F"]).shape[0])
    archive_export["size"] = archive_size
    archive_export["subset"] = _build_archive_subset_payload(state, archive_export)
    return archive_export


def _archive_execution_mode(state: NSGAIIState) -> str:
    mode = str(state.archive_mode or "off").strip().lower()
    hybrid_status = str(state.archive_hybrid_last_status or "inactive").strip().lower()
    if mode == "off":
        return "standard"
    if mode == "passive":
        return "passive_archive"
    if mode == "hybrid_survival":
        if hybrid_status == "hybrid":
            return "hybrid_survival"
        if hybrid_status == "fallback":
            return "hybrid_fallback"
        return "hybrid_requested"
    return mode


def _build_archive_diagnostics(
    state: NSGAIIState,
    archive_export: dict[str, Any] | None,
) -> dict[str, Any]:
    archive_subset = archive_export.get("subset") if isinstance(archive_export, dict) else None
    hybrid_active = str(state.archive_hybrid_last_status or "inactive").strip().lower() == "hybrid"
    return {
        "archive_mode": state.archive_mode,
        "execution_mode": _archive_execution_mode(state),
        "survival_path": "hybrid" if hybrid_active else "standard",
        "archive_present": archive_export is not None,
        "archive_size": int(archive_export["size"]) if archive_export is not None else 0,
        "archive_subset_size": int(archive_subset["size"]) if isinstance(archive_subset, dict) else 0,
        "archive_subset_selector": archive_subset.get("selector") if isinstance(archive_subset, dict) else None,
        "hybrid_status": state.archive_hybrid_last_status,
        "hybrid_fallback_reason": state.archive_hybrid_fallback_reason,
        "hybrid_split_front_mode": state.archive_hybrid_last_split_mode,
        "hybrid_split_front_reason": state.archive_hybrid_last_split_reason,
        "hybrid_generations": int(state.archive_hybrid_generations),
        "hybrid_archive_reference_generations": int(state.archive_hybrid_archive_reference_generations),
        "hybrid_local_only_generations": int(state.archive_hybrid_local_only_generations),
        "hybrid_no_split_generations": int(state.archive_hybrid_no_split_generations),
    }


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
    archive_payload = get_external_archive_payload(state)
    archive_export = _build_archive_export_payload(state, archive_payload) if archive_payload is not None else None
    should_use_archive = (
        archive_payload is not None
        and state.archive_mode not in ARCHIVE_POPULATION_RESULT_MODES
        and not wants_population_result(state)
    )

    if should_use_archive:
        result_X = np.asarray(archive_payload["X"])
        result_F = np.asarray(archive_payload["F"], dtype=float)
        archive_G_raw = archive_payload.get("G")
        result_G = None if archive_G_raw is None else np.asarray(archive_G_raw, dtype=float)
    else:
        should_filter = kernel is not None and not wants_population_result(state)
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

    if archive_export is not None:
        result["archive"] = archive_export
    result["archive_diagnostics"] = _build_archive_diagnostics(state, archive_export)

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
    archive_payload = get_external_archive_payload(state)
    if archive_payload is None:
        return None
    return dict(archive_payload)


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
    from vamos.engine.hooks.genealogy import generation_contributions, operator_success_stats

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
    G: np.ndarray | None = None,
) -> None:
    """Update external archive state and synchronize cached snapshots.

    Parameters
    ----------
    state : NSGAIIState
        Current algorithm state (modified in place).
    kernel : KernelBackend
        Kernel backend.
    X : np.ndarray | None
        Candidate decision variables to insert (defaults to state.X).
    F : np.ndarray | None
        Candidate objectives to insert (defaults to state.F).
    G : np.ndarray | None
        Candidate constraints to insert (defaults to state.G).
    """
    del kernel
    X_use = state.X if X is None else X
    F_use = state.F if F is None else F
    G_use = state.G if G is None else G
    if state.archive_manager is not None:
        state.archive_manager.update(X_use, F_use, G_use)
        if state.result_archive is state.archive_manager:
            archive_payload = get_external_archive_payload(state)
            if archive_payload is not None:
                state.archive_X = np.asarray(archive_payload["X"])
                state.archive_F = np.asarray(archive_payload["F"], dtype=float)
                archive_G_raw = archive_payload.get("G")
                state.archive_G = None if archive_G_raw is None else np.asarray(archive_G_raw, dtype=float)
            return

    if state.result_archive is not None:
        state.result_archive.update(X_use, F_use, G_use)
    archive_payload = get_external_archive_payload(state)
    if archive_payload is None:
        state.archive_X = None
        state.archive_F = None
        state.archive_G = None
        return
    state.archive_X = np.asarray(archive_payload["X"])
    state.archive_F = np.asarray(archive_payload["F"], dtype=float)
    archive_G_raw = archive_payload.get("G")
    state.archive_G = None if archive_G_raw is None else np.asarray(archive_G_raw, dtype=float)
