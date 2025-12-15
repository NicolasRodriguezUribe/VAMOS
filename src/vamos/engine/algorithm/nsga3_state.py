"""NSGA3-specific state container.

This module provides the state dataclass for NSGA-III's ask/tell interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from vamos.engine.algorithm.components.base import AlgorithmState


@dataclass
class NSGA3State(AlgorithmState):
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


def build_nsga3_result(
    state: NSGA3State,
    hv_reached: bool = False,
) -> dict[str, Any]:
    """Build NSGA-III result dictionary from state.

    Parameters
    ----------
    state : NSGA3State
        Current algorithm state.
    hv_reached : bool, optional
        Whether HV threshold was reached for early termination.

    Returns
    -------
    dict
        Result dictionary with X, F, G, reference_directions, archive data,
        and metadata.
    """
    result = {
        "X": state.X.copy(),
        "F": state.F.copy(),
        "G": state.G.copy() if state.G is not None else None,
        "reference_directions": state.ref_dirs.copy(),
        "n_eval": state.n_eval,
        "generation": state.generation,
        "hv_converged": hv_reached,
    }

    # Include archive if present
    if state.archive_X is not None and state.archive_X.size > 0:
        result["archive_X"] = state.archive_X.copy()
        result["archive_F"] = state.archive_F.copy()

    return result
