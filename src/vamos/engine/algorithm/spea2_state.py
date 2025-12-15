"""SPEA2-specific state container.

Extends AlgorithmState with SPEA2-specific fields for internal archive management.

Note: SPEA2 has two archives:
1. Internal archive (env_X, env_F, env_G): From environmental selection, used for mating
2. External archive (from base AlgorithmState): Optional crowding/hypervolume archive
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vamos.engine.algorithm.components.base import AlgorithmState


@dataclass
class SPEA2State(AlgorithmState):
    """State for SPEA2 algorithm.

    Extends AlgorithmState with SPEA2-specific fields:
    - env_X, env_F, env_G: Internal SPEA2 archive (from environmental selection)
    - env_archive_size: Target size for environmental selection
    - k_neighbors: k for k-th nearest neighbor distance calculation
    - variation: VariationPipeline for offspring generation
    - xl, xu: Bounds for offspring generation

    The base class archive_X/archive_F/archive_manager are used for the optional
    external archive (crowding/hypervolume based).
    """

    # SPEA2-specific internal archive (from environmental selection)
    env_X: np.ndarray | None = None
    env_F: np.ndarray | None = None
    env_G: np.ndarray | None = None
    env_archive_size: int = 100

    # Algorithm parameters
    k_neighbors: int | None = None

    # Variation operators (callables)
    crossover_fn: Any = None
    mutation_fn: Any = None
    xl: np.ndarray | None = None
    xu: np.ndarray | None = None


def build_spea2_result(state: SPEA2State, hv_reached: bool = False) -> dict[str, Any]:
    """Build final result dictionary from SPEA2 state.

    Parameters
    ----------
    state : SPEA2State
        The algorithm state.
    hv_reached : bool
        Whether HV termination was triggered.

    Returns
    -------
    dict
        Result dictionary with X, F, evaluations, archive, population,
        and optionally G for constrained problems.
    """
    result: dict[str, Any] = {
        "X": state.env_X,
        "F": state.env_F,
        "evaluations": state.n_eval,
        "hv_reached": hv_reached,
    }
    if state.env_G is not None and state.constraint_mode != "none":
        result["G"] = state.env_G
    result["archive"] = {"X": state.env_X, "F": state.env_F}
    result["population"] = {"X": state.X, "F": state.F}

    # Add external archive if present (from base class)
    if state.archive_X is not None:
        result["external_archive"] = {
            "X": state.archive_X,
            "F": state.archive_F,
        }

    return result
