"""
Thin facade over the core objective reduction utilities.

This module exists to preserve the analysis-facing API while delegating all
logic to ``vamos.ux.analysis.core_objective_reduction``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .core_objective_reduction import (
    ObjectiveReductionConfig,
    ObjectiveReducer as _CoreObjectiveReducer,
    reduce_objectives as _core_reduce_objectives,
)


@dataclass(frozen=True)
class ObjectiveReductionSpec:
    """Configuration wrapper used by analysis/meta-optimization code paths."""

    target_dim: int
    method: str = "correlation"
    mandatory_keep: Tuple[int, ...] = ()


class ObjectiveReducer:
    """
    Shim around the core ObjectiveReducer to keep the analysis API stable.
    """

    def __init__(self, method: str = "correlation") -> None:
        self.method = method

    def reduce(
        self,
        F: np.ndarray,
        target_dim: int,
        *,
        mandatory_keep: Iterable[int] | None = None,
    ) -> List[int]:
        _, selected = _core_reduce_objectives(
            F,
            target_dim=target_dim,
            method=self.method,
            keep_mandatory=tuple(mandatory_keep or ()),
        )
        return selected.tolist() if hasattr(selected, "tolist") else list(selected)


def reduce_objectives(
    F: np.ndarray,
    target_dim: int,
    method: str = "correlation",
    mandatory_keep: Iterable[int] | None = None,
) -> tuple[np.ndarray, List[int]]:
    """
    Convenience wrapper delegating to the core implementation.
    """
    reduced, selected = _core_reduce_objectives(
        F,
        target_dim=target_dim,
        method=method,
        keep_mandatory=tuple(mandatory_keep or ()),
    )
    selected_list = selected.tolist() if hasattr(selected, "tolist") else list(selected)
    return reduced, selected_list


__all__ = ["ObjectiveReductionSpec", "ObjectiveReducer", "reduce_objectives", "ObjectiveReductionConfig"]
