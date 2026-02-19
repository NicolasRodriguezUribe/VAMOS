"""AGE-MOEA-specific state container.

This module provides the state dataclass for AGE-MOEA's ask/tell interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vamos.engine.algorithm.components.state import AlgorithmState

if TYPE_CHECKING:
    from vamos.archive.bounded_archive import BoundedArchive
    from vamos.engine.algorithm.components.variation.pipeline import VariationPipeline


@dataclass
class AGEMOEAState(AlgorithmState):
    """State container for AGE-MOEA with ask/tell interface.

    Extends the base AlgorithmState with AGE-MOEA-specific fields.
    """

    max_evals: int = 0
    variation: VariationPipeline | None = None
    archive: BoundedArchive | None = None


def build_agemoea_result(
    state: AGEMOEAState,
    kernel: Any = None,
) -> dict[str, Any]:
    """Build AGE-MOEA result dictionary from state."""
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


__all__ = ["AGEMOEAState", "build_agemoea_result"]
