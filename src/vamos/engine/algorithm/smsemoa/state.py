"""SMSEMOA-specific state container.

This module provides the state dataclass for SMSEMOA's ask/tell interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from vamos.engine.algorithm.components.base import AlgorithmState


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
) -> dict[str, Any]:
    """Build SMSEMOA result dictionary from state.

    Parameters
    ----------
    state : SMSEMOAState
        Current algorithm state.
    hv_reached : bool, optional
        Whether HV threshold was reached for early termination.

    Returns
    -------
    dict
        Result dictionary with X, F, G, reference_point, archive data,
        and metadata.
    """
    result = {
        "X": state.X.copy(),
        "F": state.F.copy(),
        "G": state.G.copy() if state.G is not None else None,
        "reference_point": state.ref_point.copy(),
        "n_eval": state.n_eval,
        "generation": state.generation,
        "hv_converged": hv_reached,
    }

    # Include archive if present
    if state.archive_X is not None and state.archive_X.size > 0:
        result["archive_X"] = state.archive_X.copy()
        result["archive_F"] = state.archive_F.copy()

    return result
