"""SMSEMOA-specific state container.

This module provides the state dataclass for SMSEMOA's ask/tell interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

import numpy as np

from vamos.engine.algorithm.components.state import AlgorithmState


@dataclass
class SMSEMOAState(AlgorithmState):
    """State container for SMSEMOA with ask/tell interface.

    Extends the base AlgorithmState with SMSEMOA-specific fields for
    reference point management and variation operators.

    Additional Attributes
    ---------------------
    ref_point : np.ndarray
        Current reference point for hypervolume computation.
    ref_offset : float
        Offset for adaptive reference point updates.
    ref_adaptive : bool
        Whether to adaptively update reference point.
    pressure : int
        Tournament selection pressure.
    crossover_fn : Callable
        Crossover operator function.
    mutation_fn : Callable
        Mutation operator function.
    """

    ref_point: np.ndarray = field(default_factory=lambda: np.array([]))
    ref_offset: float = 0.1
    ref_adaptive: bool = True
    pressure: int = 2
    crossover_fn: Callable[[np.ndarray], np.ndarray] | None = None
    mutation_fn: Callable[[np.ndarray], np.ndarray] | None = None


def build_smsemoa_result(
    state: SMSEMOAState,
    hv_reached: bool = False,
    kernel: Any = None,
) -> dict[str, Any]:
    """Build SMSEMOA result dictionary from state.

    Parameters
    ----------
    state : SMSEMOAState
        Current algorithm state.
    hv_reached : bool, optional
        Whether HV threshold was reached for early termination.
    kernel : KernelBackend, optional
        Kernel for computing non-dominated ranking. If provided, result
        will contain only non-dominated solutions.

    Returns
    -------
    dict
        Result dictionary with X, F, G, reference_point, population, archive data,
        and metadata. X and F contain only non-dominated solutions when kernel is provided.
    """
    mode = getattr(state, "result_mode", "population")
    should_filter = kernel is not None and mode is not None and mode != "population"

    if should_filter:
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
        "reference_point": state.ref_point.copy(),
        "n_eval": state.n_eval,
        "generation": state.generation,
        "hv_converged": hv_reached,
        "population": {"X": state.X.copy(), "F": state.F.copy()},
    }

    # Include archive if present
    if state.archive_X is not None and state.archive_F is not None and state.archive_X.size > 0:
        result["archive_X"] = state.archive_X.copy()
        result["archive_F"] = state.archive_F.copy()

    return result
