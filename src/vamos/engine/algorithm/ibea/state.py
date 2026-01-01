"""IBEA-specific state container.

This module provides the state dataclass for IBEA's ask/tell interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vamos.engine.algorithm.components.state import AlgorithmState


@dataclass
class IBEAState(AlgorithmState):
    """State container for IBEA with ask/tell interface.

    Extends the base AlgorithmState with IBEA-specific fields for
    indicator and fitness tracking.

    Additional Attributes
    ---------------------
    indicator : str
        Quality indicator type ("epsilon" or "hypervolume").
    kappa : float
        Scaling factor for fitness computation.
    fitness : np.ndarray
        Current fitness values.
    pressure : int
        Tournament selection pressure.
    variation : Any
        Variation pipeline for offspring generation.
    """

    indicator: str = "epsilon"
    kappa: float = 0.05
    fitness: np.ndarray = field(default_factory=lambda: np.array([]))
    pressure: int = 2
    variation: Any = None


def build_ibea_result(
    state: IBEAState,
    hv_reached: bool = False,
    kernel: Any = None,
) -> dict[str, Any]:
    """Build IBEA result dictionary from state.

    Parameters
    ----------
    state : IBEAState
        Current algorithm state.
    hv_reached : bool, optional
        Whether HV threshold was reached for early termination.
    kernel : KernelBackend, optional
        Kernel for computing non-dominated ranking. If provided, result
        will contain only non-dominated solutions.

    Returns
    -------
    dict
        Result dictionary with X, F, G, population, archive data, and metadata.
        X and F contain only non-dominated solutions when kernel is provided.
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
        "evaluations": state.n_eval,
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


__all__ = [
    "IBEAState",
    "build_ibea_result",
]
