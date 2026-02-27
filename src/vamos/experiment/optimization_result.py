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


class TopKResult(TypedDict):
    X: NDArray[Any] | None
    F: NDArray[Any]
    indices: NDArray[np.int_]
    scores: NDArray[np.float64]
    source: str
    method: str


from vamos.foundation.quality_indicators.pareto import pareto_filter


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
            if front_F.shape[1] < 2:
                raise ValueError(f"'min_f2' requires at least 2 objectives, but this result has {front_F.shape[1]}.")
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

    def _extract_source(self, source: str) -> tuple[np.ndarray, np.ndarray | None]:
        src = str(source).strip().lower()
        if src == "result":
            F = self.F
            X = self.X
        elif src == "archive":
            archive = self.data.get("archive")
            if not isinstance(archive, Mapping):
                raise ValueError("Archive data is not available in this result.")
            F = archive.get("F")
            X = archive.get("X")
        elif src == "population":
            pop = self.data.get("population")
            if not isinstance(pop, Mapping):
                raise ValueError("Population data is not available in this result.")
            F = pop.get("F")
            X = pop.get("X")
        else:
            raise ValueError("source must be one of: result, archive, population")

        if F is None:
            raise ValueError(f"No objective values available for source='{src}'.")
        F_arr = np.asarray(F, dtype=float)
        if F_arr.ndim != 2 or F_arr.shape[0] == 0:
            raise ValueError(f"Empty or invalid objective array for source='{src}'.")
        X_arr = None if X is None else np.asarray(X)
        return F_arr, X_arr

    @staticmethod
    def _normalize_front(F: np.ndarray) -> np.ndarray:
        return (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-12)

    def _ranking_scores(
        self,
        F: np.ndarray,
        *,
        method: str,
        weights: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        key = str(method).strip().lower()
        if key == "knee":
            return np.asarray(self._normalize_front(F).sum(axis=1), dtype=float)
        if key == "min_f1":
            return np.asarray(F[:, 0], dtype=float)
        if key == "min_f2":
            if F.shape[1] < 2:
                raise ValueError(f"'min_f2' requires at least 2 objectives, but this set has {F.shape[1]}.")
            return np.asarray(F[:, 1], dtype=float)
        if key == "balanced":
            return np.asarray(self._normalize_front(F).max(axis=1), dtype=float)
        if key == "weighted_sum":
            F_norm = self._normalize_front(F)
            if weights is None:
                w = np.full(F.shape[1], 1.0 / F.shape[1], dtype=float)
            else:
                w = np.asarray(weights, dtype=float)
                if w.ndim != 1 or w.shape[0] != F.shape[1]:
                    raise ValueError(f"weights must be a 1D array with length {F.shape[1]}.")
                if np.any(w < 0.0):
                    raise ValueError("weights must be non-negative.")
                s = float(w.sum())
                if s <= 0.0:
                    raise ValueError("weights must sum to a positive value.")
                w = w / s
            return np.asarray(F_norm @ w, dtype=float)
        raise ValueError("Unknown method. Use: knee, min_f1, min_f2, balanced, weighted_sum, crowding")

    def top_k(
        self,
        k: int = 100,
        *,
        source: str = "archive",
        method: str = "knee",
        nondominated_only: bool = True,
        weights: NDArray[np.float64] | None = None,
    ) -> TopKResult:
        """
        Rank solutions and return the top-k rows from a selected source.

        Args:
            k: Number of rows to return (capped to available rows).
            source: One of ``result``, ``archive``, or ``population``.
            method: Ranking method: ``knee``, ``min_f1``, ``min_f2``,
                ``balanced``, ``weighted_sum``, or ``crowding``.
            nondominated_only: If True, rank only the first Pareto front.
            weights: Optional weights for ``weighted_sum`` ranking.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        F_src, X_src = self._extract_source(source)
        idx_src = np.arange(F_src.shape[0], dtype=int)

        if nondominated_only:
            front = pareto_filter(F_src, return_indices=True)
            if front is None:
                raise ValueError("No solutions available after Pareto filtering.")
            F_rank, idx_rank = front
            idx_rank = np.asarray(idx_rank, dtype=int)
            if F_rank.size == 0:
                raise ValueError("No solutions available after Pareto filtering.")
        else:
            F_rank = F_src
            idx_rank = idx_src

        key = str(method).strip().lower()
        if key == "crowding":
            from vamos.engine.algorithm.components.archive import (
                _single_front_crowding,
                select_top_k_crowding,
            )

            k_eff = min(int(k), F_rank.shape[0])
            selected_local = select_top_k_crowding(F_rank, k_eff)
            selected_idx = idx_rank[selected_local]
            selected_F = F_src[selected_idx]
            selected_X = X_src[selected_idx] if X_src is not None else None
            selected_scores = np.asarray(
                _single_front_crowding(F_rank)[selected_local], dtype=float
            )
            return {
                "X": selected_X,
                "F": selected_F,
                "indices": np.asarray(selected_idx, dtype=int),
                "scores": selected_scores,
                "source": str(source).strip().lower(),
                "method": "crowding",
            }

        scores = self._ranking_scores(F_rank, method=method, weights=weights)
        order = np.argsort(scores)
        k_eff = min(int(k), order.shape[0])
        order = order[:k_eff]
        selected_idx = idx_rank[order]
        selected_F = F_src[selected_idx]
        selected_X = X_src[selected_idx] if X_src is not None else None
        selected_scores = np.asarray(scores[order], dtype=float)

        return {
            "X": selected_X,
            "F": selected_F,
            "indices": np.asarray(selected_idx, dtype=int),
            "scores": selected_scores,
            "source": str(source).strip().lower(),
            "method": str(method).strip().lower(),
        }

    def top_k_report(
        self,
        k: int = 100,
        *,
        source: str = "archive",
        method: str = "knee",
        nondominated_only: bool = True,
        weights: NDArray[np.float64] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return ranked top-k rows as serializable records for reporting/export.
        """
        top = self.top_k(
            k=k,
            source=source,
            method=method,
            nondominated_only=nondominated_only,
            weights=weights,
        )
        F = np.asarray(top["F"], dtype=float)
        X = top["X"]
        idx = np.asarray(top["indices"], dtype=int)
        scores = np.asarray(top["scores"], dtype=float)
        rows: list[dict[str, Any]] = []
        for rank in range(F.shape[0]):
            row: dict[str, Any] = {
                "rank": rank + 1,
                "index": int(idx[rank]),
                "score": float(scores[rank]),
                "source": top["source"],
                "method": top["method"],
            }
            for j in range(F.shape[1]):
                row[f"f{j + 1}"] = float(F[rank, j])
            if X is not None:
                X_arr = np.asarray(X)
                for j in range(X_arr.shape[1]):
                    value = X_arr[rank, j]
                    row[f"x{j + 1}"] = value.item() if hasattr(value, "item") else value
            rows.append(row)
        return rows

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


__all__ = ["BestResult", "TopKResult", "OptimizationResult"]
