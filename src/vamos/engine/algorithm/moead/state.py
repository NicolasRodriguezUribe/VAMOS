"""
MOEA/D State container and result building.

This module provides the MOEADState dataclass that holds all mutable state
for the MOEA/D algorithm, similar to NSGAIIState.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from vamos.engine.algorithm.components.base import AlgorithmState

if TYPE_CHECKING:
    from vamos.engine.algorithm.components.archive import CrowdingDistanceArchive, HypervolumeArchive
    from vamos.engine.algorithm.components.termination import HVTracker
    from vamos.ux.analytics.genealogy import GenealogyTracker


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
    neighbors: np.ndarray = field(default_factory=lambda: np.array([]))
    ideal: np.ndarray = field(default_factory=lambda: np.array([]))
    aggregator: Callable | None = None
    neighbor_size: int = 20
    delta: float = 0.9
    replace_limit: int = 2

    # Variation operators (as callables for flexibility across encodings)
    crossover_fn: Callable | None = None
    mutation_fn: Callable | None = None

    # Bounds for integer/real encodings
    xl: np.ndarray | None = None
    xu: np.ndarray | None = None

    # Pending offspring tracking for ask/tell
    pending_active_indices: np.ndarray | None = None
    pending_parent_pairs: np.ndarray | None = None


def build_moead_result(
    state: MOEADState,
    hv_reached: bool = False,
) -> dict[str, Any]:
    """
    Build the result dictionary from MOEA/D state.

    Parameters
    ----------
    state : MOEADState
        Current algorithm state.
    hv_reached : bool
        Whether HV threshold was reached.

    Returns
    -------
    dict[str, Any]
        Result dictionary with X, F, weights, evaluations, and optional archive.
    """
    result: dict[str, Any] = {
        "X": state.X,
        "F": state.F,
        "weights": state.weights,
        "evaluations": state.n_eval,
        "hv_reached": hv_reached,
    }

    if state.G is not None:
        result["G"] = state.G

    # Add archive contents
    if state.archive_manager is not None:
        arch_X, arch_F = state.archive_manager.contents()
        result["archive"] = {"X": arch_X, "F": arch_F}
    elif state.archive_X is not None and state.archive_F is not None:
        result["archive"] = {"X": state.archive_X, "F": state.archive_F}

    return result


__all__ = [
    "MOEADState",
    "build_moead_result",
]
