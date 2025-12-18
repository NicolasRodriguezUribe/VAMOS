"""NSGA3-specific state container.

This module provides the state dataclass for NSGA-III's ask/tell interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from vamos.engine.algorithm.components.base import AlgorithmState


@dataclass
class NSGAIIIState(AlgorithmState):
    """State container for NSGA-III with ask/tell interface.

    Extends the base AlgorithmState with NSGA-III-specific fields for
    reference directions and variation operators.

    Additional Attributes
    ---------------------
    ref_dirs : np.ndarray
        Reference directions for diversity maintenance.
    ref_dirs_norm : np.ndarray
        Normalized reference directions.
    pressure : int
        Tournament selection pressure.
    crossover_fn : Callable
        Crossover operator function.
    mutation_fn : Callable
        Mutation operator function.
    """

    ref_dirs: np.ndarray = field(default_factory=lambda: np.array([]))
    ref_dirs_norm: np.ndarray = field(default_factory=lambda: np.array([]))
    pressure: int = 2
    crossover_fn: Callable[[np.ndarray], np.ndarray] | None = None
    mutation_fn: Callable[[np.ndarray], np.ndarray] | None = None


def build_nsgaiii_result(
    state: NSGAIIIState,
    hv_reached: bool = False,
    kernel: Any = None,
) -> dict[str, Any]:
    """Build NSGA-III result dictionary from state.

    Parameters
    ----------
    state : NSGAIIIState
        Current algorithm state.
    hv_reached : bool, optional
        Whether HV threshold was reached for early termination.
    kernel : KernelBackend, optional
        Kernel for computing non-dominated ranking. If provided, result
        will contain only non-dominated solutions.

    Returns
    -------
    dict
        Result dictionary with X, F, G, reference_directions, population, archive data,
        and metadata. X and F contain only non-dominated solutions when kernel is provided.
    """
    # Filter to non-dominated solutions only
    if kernel is not None:
        try:
            ranks, _ = kernel.nsga2_ranking(state.F)
            nd_mask = ranks == ranks.min(initial=0)
            result_X = state.X[nd_mask].copy()
            result_F = state.F[nd_mask].copy()
            result_G = state.G[nd_mask].copy() if state.G is not None else None
        except (ValueError, IndexError):
            result_X = state.X.copy()
            result_F = state.F.copy()
            result_G = state.G.copy() if state.G is not None else None
    else:
        result_X = state.X.copy()
        result_F = state.F.copy()
        result_G = state.G.copy() if state.G is not None else None

    result = {
        "X": result_X,
        "F": result_F,
        "G": result_G,
        "reference_directions": state.ref_dirs.copy(),
        "n_eval": state.n_eval,
        "generation": state.generation,
        "hv_converged": hv_reached,
        "population": {"X": state.X.copy(), "F": state.F.copy()},
    }

    # Include archive if present
    if state.archive_X is not None and state.archive_X.size > 0:
        result["archive_X"] = state.archive_X.copy()
        result["archive_F"] = state.archive_F.copy()

    return result
