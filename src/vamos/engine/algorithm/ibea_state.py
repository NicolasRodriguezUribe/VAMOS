"""IBEA-specific state container.

This module provides the state dataclass for IBEA's ask/tell interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from vamos.engine.algorithm.components.base import AlgorithmState


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
) -> dict[str, Any]:
    """Build IBEA result dictionary from state.

    Parameters
    ----------
    state : IBEAState
        Current algorithm state.
    hv_reached : bool, optional
        Whether HV threshold was reached for early termination.

    Returns
    -------
    dict
        Result dictionary with X, F, G, archive data, and metadata.
    """
    result = {
        "X": state.X.copy(),
        "F": state.F.copy(),
        "G": state.G.copy() if state.G is not None else None,
        "evaluations": state.n_eval,
        "n_eval": state.n_eval,
        "generation": state.generation,
        "hv_converged": hv_reached,
    }

    # Include archive if present
    if state.archive_X is not None and state.archive_X.size > 0:
        result["archive_X"] = state.archive_X.copy()
        result["archive_F"] = state.archive_F.copy()

    return result
