"""RVEA-specific state container.

This module provides the state dataclass for RVEA's ask/tell interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.state import AlgorithmState

if TYPE_CHECKING:
    from vamos.archive.bounded_archive import BoundedArchive
    from vamos.engine.algorithm.components.variation.pipeline import VariationPipeline


@dataclass
class RVEAState(AlgorithmState):
    """State container for RVEA with ask/tell interface.

    Extends the base AlgorithmState with RVEA-specific fields for
    reference vectors and adaptation.
    """

    max_evals: int = 0
    max_gen: int = 1
    variation: VariationPipeline | None = None
    archive: BoundedArchive | None = None
    ref_dirs: np.ndarray = field(default_factory=lambda: np.array([]))
    V: np.ndarray = field(default_factory=lambda: np.array([]))
    gamma: np.ndarray = field(default_factory=lambda: np.array([]))
    ideal: np.ndarray = field(default_factory=lambda: np.array([]))
    nadir: np.ndarray | None = None
    alpha: float = 2.0
    adapt_interval: int | None = None
    n_obj: int = 2


def build_rvea_result(
    state: RVEAState,
    kernel: Any = None,
) -> dict[str, Any]:
    """Build RVEA result dictionary from state."""
    mode = getattr(state, "result_mode", "non_dominated")

    if mode != "population" and kernel is not None:
        try:
            ranks, _ = kernel.nsga2_ranking(state.F)
            nd_mask = ranks == 0
            result_X = state.X[nd_mask].copy()
            result_F = state.F[nd_mask].copy()
        except (ValueError, IndexError):
            result_X = state.X.copy()
            result_F = state.F.copy()
    else:
        result_X = state.X.copy()
        result_F = state.F.copy()

    result: dict[str, Any] = {
        "X": result_X,
        "F": result_F,
        "n_eval": state.n_eval,
        "n_gen": state.generation,
        "population": {"X": state.X.copy(), "F": state.F.copy()},
    }
    if state.archive is not None:
        result["archive"] = {"X": state.archive.X, "F": state.archive.F}
    return result


__all__ = ["RVEAState", "build_rvea_result"]
