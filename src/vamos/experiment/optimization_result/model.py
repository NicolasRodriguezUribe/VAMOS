from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypedDict, overload

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.quality_indicators.pareto import pareto_filter

from .ranking import top_k as rank_top_k
from .ranking import top_k_report as build_top_k_report


class BestResult(TypedDict):
    X: NDArray[Any] | None
    F: NDArray[Any]
    index: int
    front_index: int


class TopKResult(TypedDict):
    X: NDArray[Any] | None
    F: NDArray[Any]
    indices: NDArray[np.int_]
    scores: NDArray[np.float64]
    source: str
    method: str


class OptimizationResult:
    """
    Container returned by optimize() with Pareto front data and selection helpers.

    Use `vamos.ux.api` for summaries, plotting, and export helpers.
    """

    F: NDArray[Any] | None
    X: NDArray[Any] | None
    data: dict[str, Any]
    meta: dict[str, Any]

    def __init__(self, payload: Mapping[str, Any], *, meta: Mapping[str, Any] | None = None):
        self.F = payload.get("F")
        self.X = payload.get("X")
        self.data = dict(payload)
        self.meta = dict(meta or {})

    def __len__(self) -> int:
        return len(self.F) if self.F is not None else 0

    def __repr__(self) -> str:
        n_sol = len(self)
        n_obj = self.F.shape[1] if self.F is not None and len(self.F) > 0 else 0
        return f"OptimizationResult({n_sol} solutions, {n_obj} objectives)"

    @property
    def n_objectives(self) -> int:
        return self.F.shape[1] if self.F is not None and len(self.F) > 0 else 0

    @overload
    def front(self, *, return_indices: Literal[False] = False) -> np.ndarray | None: ...

    @overload
    def front(self, *, return_indices: Literal[True]) -> tuple[np.ndarray, np.ndarray]: ...

    def front(self, *, return_indices: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
        if return_indices:
            return pareto_filter(self.F, return_indices=True)
        return pareto_filter(self.F, return_indices=False)

    def best(self, method: str = "knee") -> BestResult:
        if self.F is None or len(self.F) == 0:
            raise ValueError("No solutions available")

        front = self.front(return_indices=True)
        if front is None:
            raise ValueError("No solutions available")
        front_F, front_idx = front
        if len(front_F) == 0:
            raise ValueError("No solutions available")

        if method == "knee":
            F_norm = (front_F - front_F.min(axis=0)) / (np.ptp(front_F, axis=0) + 1e-12)
            front_pos = int(np.argmin(F_norm.sum(axis=1)))
        elif method == "min_f1":
            front_pos = int(np.argmin(front_F[:, 0]))
        elif method == "min_f2":
            if front_F.shape[1] < 2:
                raise ValueError(f"'min_f2' requires at least 2 objectives, but this result has {front_F.shape[1]}.")
            front_pos = int(np.argmin(front_F[:, 1]))
        elif method == "balanced":
            F_norm = (front_F - front_F.min(axis=0)) / (np.ptp(front_F, axis=0) + 1e-12)
            front_pos = int(np.argmin(F_norm.max(axis=1)))
        else:
            raise ValueError(f"Unknown method '{method}'. Use: knee, min_f1, min_f2, balanced")

        idx = int(front_idx[front_pos])
        return {
            "X": self.X[idx] if self.X is not None else None,
            "F": self.F[idx],
            "index": idx,
            "front_index": front_pos,
        }

    def top_k(
        self,
        k: int = 100,
        *,
        source: str = "archive",
        method: str = "knee",
        nondominated_only: bool = True,
        weights: NDArray[np.float64] | None = None,
    ) -> TopKResult:
        return rank_top_k(
            self,
            k=k,
            source=source,
            method=method,
            nondominated_only=nondominated_only,
            weights=weights,
        )

    def top_k_report(
        self,
        k: int = 100,
        *,
        source: str = "archive",
        method: str = "knee",
        nondominated_only: bool = True,
        weights: NDArray[np.float64] | None = None,
    ) -> list[dict[str, Any]]:
        return build_top_k_report(
            result=self,
            k=k,
            source=source,
            method=method,
            nondominated_only=nondominated_only,
            weights=weights,
        )

    def explain_defaults(self) -> dict[str, object]:
        explained: dict[str, object] = {}
        resolved = self.meta.get("resolved_config")
        sources = self.meta.get("default_sources")
        if resolved is not None:
            explained["resolved_config"] = resolved
        if sources is not None:
            explained["default_sources"] = sources
        return explained


__all__ = ["BestResult", "OptimizationResult", "TopKResult"]
