from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypedDict, overload

import numpy as np
from numpy.typing import NDArray


class BestResult(TypedDict):
    X: NDArray[Any] | None
    F: NDArray[Any]
    index: int
    front_index: int


from vamos.foundation.metrics.pareto import pareto_filter


class OptimizationResult:
    """
    Container returned by optimize() with Pareto front data and selection helpers.

    Use `vamos.ux.api` for summaries, plotting, and export helpers.

    Attributes:
        F: Objective values array (n_solutions, n_objectives)
        X: Decision variables array (n_solutions, n_variables), may be None
        data: Full result dictionary with all fields
        meta: Metadata such as algorithm, engine, seed, and termination

    Examples:
        >>> result = optimize("zdt1", algorithm="nsgaii", max_evaluations=1000)
        >>> from vamos.ux.api import log_result_summary, plot_result_front
        >>> log_result_summary(result)  # Log quick overview
        >>> plot_result_front(result)  # Visualize Pareto front
        >>> best = result.best("knee")  # Select a solution
        >>> from vamos.ux.api import result_to_dataframe
        >>> df = result_to_dataframe(result)  # Export to pandas
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
        """Number of solutions in the result."""
        return len(self.F) if self.F is not None else 0

    def __repr__(self) -> str:
        n_sol = len(self)
        n_obj = self.F.shape[1] if self.F is not None and len(self.F) > 0 else 0
        return f"OptimizationResult({n_sol} solutions, {n_obj} objectives)"

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""
        return self.F.shape[1] if self.F is not None and len(self.F) > 0 else 0

    @overload
    def front(self, *, return_indices: Literal[False] = False) -> np.ndarray | None: ...

    @overload
    def front(self, *, return_indices: Literal[True]) -> tuple[np.ndarray, np.ndarray]: ...

    def front(self, *, return_indices: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
        """
        Return non-dominated solutions (first Pareto front).

        Args:
            return_indices: When True, also return indices of the front in F.
        """
        if return_indices:
            return pareto_filter(self.F, return_indices=True)
        return pareto_filter(self.F, return_indices=False)

    def best(self, method: str = "knee") -> BestResult:
        """
        Select a single 'best' solution from the Pareto front.

        Args:
            method: Selection method - 'knee' (default), 'min_f1', 'min_f2', 'balanced'

        Returns:
            Dictionary with 'X' (decision vars), 'F' (objectives),
            'index' (position in original F/X), and 'front_index' (position in the front)
        """
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
            front_pos = int(np.argmin(front_F[:, 1]))
        elif method == "balanced":
            F_norm = (front_F - front_F.min(axis=0)) / (np.ptp(front_F, axis=0) + 1e-12)
            front_pos = int(np.argmin(F_norm.max(axis=1)))
        else:
            raise ValueError(f"Unknown method '{method}'. Use: knee, min_f1, min_f2, balanced")

        idx = int(front_idx[front_pos])
        X_sel = self.X[idx] if self.X is not None else None
        return {
            "X": X_sel,
            "F": self.F[idx],
            "index": idx,
            "front_index": front_pos,
        }

    def explain_defaults(self) -> dict[str, object]:
        """
        Return resolved configuration and default sources, if available.

        The unified optimize() API records metadata about which settings were
        inferred vs provided. This method surfaces that metadata in a stable
        dict form for reporting or debugging.
        """
        explained: dict[str, object] = {}
        resolved = self.meta.get("resolved_config")
        sources = self.meta.get("default_sources")
        if resolved is not None:
            explained["resolved_config"] = resolved
        if sources is not None:
            explained["default_sources"] = sources
        return explained


__all__ = ["BestResult", "OptimizationResult"]
