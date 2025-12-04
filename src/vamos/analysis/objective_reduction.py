"""
Objective reduction utilities for many-objective analysis.

This wraps the core reducer (``vamos.objective_reduction``) with a small façade
for post-hoc workflows and optional use in meta-optimization. It does not alter
core algorithm kernels; reduction is always optional.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ObjectiveReductionSpec:
    """Configuration for optional objective reduction."""

    target_dim: int
    method: str = "correlation"
    mandatory_keep: Tuple[int, ...] = ()


class ObjectiveReducer:
    """
    Thin façade around the core ObjectiveReducer to reduce objectives from F.
    """

    def __init__(self, method: str = "correlation") -> None:
        """
        Args:
            method: One of {"correlation", "angle", "hybrid"}.
        """
        self.method = method

    def reduce(
        self,
        F: np.ndarray,
        target_dim: int,
        *,
        mandatory_keep: Iterable[int] | None = None,
    ) -> List[int]:
        from vamos.objective_reduction import reduce_objectives as _reduce_objectives

        keep = tuple(mandatory_keep or ())
        _, selected = _reduce_objectives(
            F,
            target_dim=target_dim,
            method=self.method,
            keep_mandatory=keep,
        )
        return list(selected.tolist() if hasattr(selected, "tolist") else selected)


def reduce_objectives(
    F: np.ndarray,
    target_dim: int,
    method: str = "correlation",
    mandatory_keep: Iterable[int] | None = None,
) -> tuple[np.ndarray, List[int]]:
    """
    Convenience: reduce F to target_dim objectives.
    Returns (F_reduced, selected_indices).
    """
    reducer = ObjectiveReducer(method=method)
    selected = reducer.reduce(F, target_dim, mandatory_keep=mandatory_keep)
    return F[:, selected], selected
