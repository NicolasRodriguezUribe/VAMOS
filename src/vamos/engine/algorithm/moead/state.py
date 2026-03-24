"""
MOEA/D State container and result building.

This module provides the MOEADState dataclass that holds all mutable state
for the MOEA/D algorithm, similar to NSGAIIState.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.results import get_external_archive_contents, wants_population_result
from vamos.engine.algorithm.components.state import AlgorithmState

if TYPE_CHECKING:
    from vamos.engine.adaptation.online_control import HierarchicalAction, OnlineControlController, OperatorFamily, SearchState
    from vamos.engine.adaptation.online_control.adapters.base import VariationDescriptor
    from vamos.engine.adaptation.online_control.adapters.moead import MOEADOnlineControlAdapter


@dataclass
class MOEADState(AlgorithmState):
    """
    Mutable state container for MOEA/D algorithm.

    Extends AlgorithmState with MOEA/D-specific fields for weight vectors,
    neighborhoods, aggregation, and subproblem management.

    Attributes
    ----------
    weights : np.ndarray
        Weight vectors for decomposition, shape (pop_size, n_obj).
    neighbors : np.ndarray
        Neighborhood indices for each subproblem, shape (pop_size, T).
    ideal : np.ndarray
        Ideal point (minimum objectives seen), shape (n_obj,).
    aggregator : Callable
        Aggregation function (tchebycheff, weighted_sum, pbi, etc.).
    neighbor_size : int
        Neighborhood size (T parameter).
    delta : float
        Probability of selecting from neighborhood vs. whole population.
    replace_limit : int
        Maximum number of replacements per offspring.
    track_genealogy : bool
        Whether to track genealogy information.
    genealogy_tracker : GenealogyTracker | None
        Tracker for lineage information.
    ids : np.ndarray | None
        Individual IDs for genealogy tracking.
    """

    # MOEA/D-specific fields
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    weights_safe: np.ndarray = field(default_factory=lambda: np.array([]))
    weights_unit: np.ndarray = field(default_factory=lambda: np.array([]))
    neighbors: np.ndarray = field(default_factory=lambda: np.array([]))
    ideal: np.ndarray = field(default_factory=lambda: np.array([]))
    aggregator: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None
    aggregation_id: int = -1
    aggregation_theta: float = 5.0
    aggregation_rho: float = 0.001
    neighbor_size: int = 20
    delta: float = 0.9
    replace_limit: int = 2
    batch_size: int = 1
    subproblem_order: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    subproblem_cursor: int = 0

    # Cached constraint violation (sum of positive parts), when constraints are enabled.
    cv: np.ndarray | None = None

    # Variation operators (as callables for flexibility across encodings)
    crossover_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None
    mutation_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None

    # Bounds for integer/real encodings
    xl: np.ndarray | None = None
    xu: np.ndarray | None = None
    encoding: str = "real"
    problem: Any | None = None
    max_evaluations: int = 0
    quality_indicator: float = 0.0
    best_quality_indicator: float | None = None
    stagnant_steps: int = 0
    current_cross_method: str = ""
    current_mutation_method: str = ""
    current_cross_is_de: bool = False
    base_variation_config: dict[str, Any] = field(default_factory=dict)

    # Pending offspring tracking for ask/tell
    pending_active_indices: np.ndarray | None = None
    pending_parent_pairs: np.ndarray | None = None
    pending_use_neighbors: np.ndarray | None = None
    online_control_controller: OnlineControlController | None = None
    online_control_adapter: MOEADOnlineControlAdapter | None = None
    online_control_families: tuple[OperatorFamily, ...] = field(default_factory=tuple)
    current_online_descriptor: VariationDescriptor | None = None
    pending_online_descriptor: VariationDescriptor | None = None
    pending_online_action: HierarchicalAction | None = None
    pending_search_state: SearchState | None = None
    pending_online_overhead_ms: float | None = None
    online_control_runtime_profile: dict[str, float] = field(default_factory=dict)


def build_moead_result(
    state: MOEADState,
    hv_reached: bool = False,
    kernel: Any = None,
) -> dict[str, Any]:
    """
    Build the result dictionary from MOEA/D state.

    Parameters
    ----------
    state : MOEADState
        Current algorithm state.
    hv_reached : bool
        Whether HV threshold was reached.
    kernel : KernelBackend, optional
        Kernel for computing non-dominated ranking. If provided, result
        will contain only non-dominated solutions.

    Returns
    -------
    dict[str, Any]
        Result dictionary with X, F, weights, evaluations, population, and optional archive.
        X and F contain only non-dominated solutions when kernel is provided.
    """
    archive_contents = get_external_archive_contents(state)
    if archive_contents is not None and not wants_population_result(state):
        result_X, result_F = archive_contents
        result_G = None
    else:
        should_filter = kernel is not None and not wants_population_result(state)
        if should_filter:
            try:
                ranks, _ = kernel.nsga2_ranking(state.F)
                nd_mask = ranks == ranks.min(initial=0)
                result_X = state.X[nd_mask]
                result_F = state.F[nd_mask]
                result_G = state.G[nd_mask] if state.G is not None else None
            except (ValueError, IndexError):
                result_X, result_F, result_G = state.X, state.F, state.G
        else:
            result_X, result_F, result_G = state.X, state.F, state.G

    result: dict[str, Any] = {
        "X": result_X,
        "F": result_F,
        "weights": state.weights,
        "evaluations": state.n_eval,
        "hv_reached": hv_reached,
        "population": {"X": state.X, "F": state.F},
    }

    if result_G is not None:
        result["G"] = result_G

    if archive_contents is not None:
        arch_X, arch_F = archive_contents
        result["archive"] = {"X": arch_X, "F": arch_F}
    if state.online_control_controller is not None:
        payload = state.online_control_controller.result_payload()
        runtime_profile = {key: float(value) for key, value in state.online_control_runtime_profile.items()}
        payload["runtime_profile"] = runtime_profile
        run_summary = payload.get("run_summary")
        if isinstance(run_summary, dict):
            updated_run_summary = dict(run_summary)
            updated_run_summary["runtime_profile"] = runtime_profile
            updated_run_summary.update(runtime_profile)
            payload["run_summary"] = updated_run_summary
        result["online_control"] = payload

    return result


__all__ = [
    "MOEADState",
    "build_moead_result",
]
