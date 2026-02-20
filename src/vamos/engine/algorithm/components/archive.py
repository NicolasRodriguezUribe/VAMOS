from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np

try:  # pragma: no cover - optional dependency
    import moocore as _moocore
except ImportError:  # pragma: no cover - optional dependency
    _moocore = None

from vamos.foundation.constraints.utils import compute_violation, is_feasible
from vamos.foundation.kernel.numpy_backend import _compute_crowding
from vamos.foundation.metrics.pareto import pareto_filter

DeduplicateIn = Literal["objective", "decision", "both"]
_HV_FALLBACK_WARNED = False


def _single_front_crowding(F: np.ndarray) -> np.ndarray:
    """Crowding distance for a single nondominated front."""
    if F.shape[0] == 0:
        return np.empty(0, dtype=float)
    fronts = [list(range(F.shape[0]))]
    return _compute_crowding(F, fronts)


def _hv_contributions(F: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Compute hypervolume contribution of each point.

    Uses moocore when available; otherwise falls back to crowding distance and
    emits a one-time warning.
    """
    global _HV_FALLBACK_WARNED
    if F.shape[0] == 0:
        return np.empty(0, dtype=float)
    if _moocore is not None:
        return np.asarray(_moocore.hv_contributions(F, ref=ref), dtype=float)
    if not _HV_FALLBACK_WARNED:
        warnings.warn(
            "Hypervolume contributions requested but 'moocore' is not installed; "
            "falling back to crowding distance.",
            UserWarning,
            stacklevel=2,
        )
        _HV_FALLBACK_WARNED = True
    return _single_front_crowding(F)


def _nondominated_mask(F: np.ndarray) -> np.ndarray:
    if F.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if _moocore is not None:
        return np.asarray(_moocore.is_nondominated(F), dtype=bool)
    _, nd_idx = pareto_filter(F, return_indices=True)
    mask = np.zeros(F.shape[0], dtype=bool)
    mask[nd_idx] = True
    return mask


def _unique_rows_with_tolerance(values: np.ndarray, tol: float) -> np.ndarray:
    n = int(values.shape[0])
    if n <= 1:
        return np.arange(n, dtype=int)
    if tol < 0.0:
        return np.arange(n, dtype=int)
    if tol == 0.0:
        _, unique_idx = np.unique(values, axis=0, return_index=True)
        unique_idx.sort()
        return np.asarray(unique_idx, dtype=int)

    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        diff = np.abs(values[i + 1 :] - values[i])
        if diff.size == 0:
            continue
        dup_mask = np.all(diff <= tol, axis=1)
        if dup_mask.any():
            keep[i + 1 :][dup_mask] = False
    return np.flatnonzero(keep)


def _subset_arrays(
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray | None,
    idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if idx.size == 0:
        return X[:0], F[:0], G[:0] if G is not None else None
    return X[idx], F[idx], G[idx] if G is not None else None


class _BaseArchive:
    """
    Base class for bounded external archives.

    Update is batch-based: merge existing + incoming, feasibility filter,
    non-dominated extraction, deduplication, then (if needed) truncation.
    """

    def __init__(
        self,
        capacity: int,
        n_var: int,
        n_obj: int,
        dtype: Any,
        *,
        truncate_size: int | None = None,
        objective_tolerance: float = 1e-10,
        deduplicate_in: DeduplicateIn = "objective",
        decision_tolerance: float = 1e-32,
        n_con: int | None = None,
    ) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("archive capacity must be positive.")

        self.truncate_size = self.capacity if truncate_size is None else int(truncate_size)
        if self.truncate_size <= 0:
            raise ValueError("truncate_size must be positive.")
        if self.truncate_size > self.capacity:
            raise ValueError("truncate_size must be <= capacity.")

        if objective_tolerance < 0.0:
            raise ValueError("objective_tolerance must be >= 0.")
        if decision_tolerance < 0.0:
            raise ValueError("decision_tolerance must be >= 0.")

        if deduplicate_in not in {"objective", "decision", "both"}:
            raise ValueError("deduplicate_in must be 'objective', 'decision', or 'both'.")

        self._dtype = np.dtype(dtype)
        self._n_var = int(n_var)
        self._n_obj = int(n_obj)
        self._objective_tolerance = float(objective_tolerance)
        self._decision_tolerance = float(decision_tolerance)
        self._deduplicate_in = deduplicate_in

        self._n_con = int(n_con) if n_con is not None else None
        if self._n_con is not None and self._n_con <= 0:
            raise ValueError("n_con must be positive when provided.")

        storage_rows = max(self.capacity + 1, self.truncate_size + 1)
        self._X = np.empty((storage_rows, self._n_var), dtype=self._dtype)
        self._F = np.empty((storage_rows, self._n_obj), dtype=float)
        self._G = np.empty((storage_rows, self._n_con), dtype=float) if self._n_con is not None else None
        self._size = 0

    def update(
        self,
        population_X: np.ndarray,
        population_F: np.ndarray,
        population_G: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pop_X = np.asarray(population_X, dtype=self._dtype, order="C")
        pop_F = np.asarray(population_F, dtype=float, order="C")
        if pop_X.ndim != 2:
            raise ValueError("population_X must be a 2D array.")
        if pop_F.ndim != 2:
            raise ValueError("population_F must be a 2D array.")
        if pop_X.shape[0] != pop_F.shape[0]:
            raise ValueError("population_X and population_F must have the same number of rows.")
        if pop_X.shape[1] != self._n_var:
            raise ValueError(f"population_X has {pop_X.shape[1]} columns, expected {self._n_var}.")
        if pop_F.shape[1] != self._n_obj:
            raise ValueError(f"population_F has {pop_F.shape[1]} columns, expected {self._n_obj}.")
        if pop_X.shape[0] == 0:
            return self._snapshot()

        pop_G: np.ndarray | None = None
        if population_G is not None:
            pop_G = np.asarray(population_G, dtype=float, order="C")
            if pop_G.ndim != 2:
                raise ValueError("population_G must be a 2D array when provided.")
            if pop_G.shape[0] != pop_X.shape[0]:
                raise ValueError("population_G must have the same number of rows as population_X.")

        X_comb, F_comb, G_comb = self._merge_with_existing(pop_X, pop_F, pop_G)
        X_work, F_work, G_work = self._apply_feasibility_filter(X_comb, F_comb, G_comb)
        if F_work.shape[0] == 0:
            self._replace_contents(X_work, F_work, G_work)
            return self._snapshot()

        nd_mask = _nondominated_mask(F_work)
        X_nd = X_work[nd_mask]
        F_nd = F_work[nd_mask]
        G_nd = G_work[nd_mask] if G_work is not None else None

        X_nd, F_nd, G_nd = self._dedupe(X_nd, F_nd, G_nd)

        if F_nd.shape[0] > self.capacity:
            keep_idx = np.asarray(self._select_subset(F_nd, self.truncate_size, G_nd), dtype=int)
            if keep_idx.ndim != 1:
                raise ValueError("_select_subset() must return a 1D index array.")
            if keep_idx.size != self.truncate_size:
                raise ValueError(
                    f"_select_subset() returned {keep_idx.size} indices, expected {self.truncate_size}."
                )
            X_nd, F_nd, G_nd = _subset_arrays(X_nd, F_nd, G_nd, keep_idx)

        self._replace_contents(X_nd, F_nd, G_nd)
        return self._snapshot()

    def contents(self) -> tuple[np.ndarray, np.ndarray]:
        return self._snapshot()

    def _snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        return self._X[: self._size].copy(), self._F[: self._size].copy()

    def _merge_with_existing(
        self,
        pop_X: np.ndarray,
        pop_F: np.ndarray,
        pop_G: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self._size == 0:
            return pop_X, pop_F, pop_G

        total = self._size + pop_X.shape[0]
        X_comb = np.empty((total, self._n_var), dtype=self._dtype)
        F_comb = np.empty((total, self._n_obj), dtype=float)

        X_comb[: self._size] = self._X[: self._size]
        F_comb[: self._size] = self._F[: self._size]
        X_comb[self._size :] = pop_X
        F_comb[self._size :] = pop_F

        if self._G is None and pop_G is None:
            return X_comb, F_comb, None

        if pop_G is not None:
            n_con = pop_G.shape[1]
        else:
            assert self._G is not None
            n_con = self._G.shape[1]
        if self._G is not None and self._G.shape[1] != n_con:
            raise ValueError("Constraint column count mismatch with existing archive contents.")
        if pop_G is not None and pop_G.shape[1] != n_con:
            raise ValueError("population_G has an incompatible number of constraint columns.")

        G_comb = np.zeros((total, n_con), dtype=float)
        if self._G is not None and self._size > 0:
            G_comb[: self._size] = self._G[: self._size]
        if pop_G is not None:
            G_comb[self._size :] = pop_G
        return X_comb, F_comb, G_comb

    @staticmethod
    def _apply_feasibility_filter(
        X: np.ndarray,
        F: np.ndarray,
        G: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if G is None or G.shape[0] == 0:
            return X, F, G

        feas = is_feasible(G, n=G.shape[0])
        if feas.any():
            keep = feas
            return X[keep], F[keep], G[keep]

        cv = compute_violation(G, n=G.shape[0])
        min_cv = float(np.min(cv))
        keep = cv <= (min_cv + 1e-12)
        return X[keep], F[keep], G[keep]

    def _dedupe(
        self,
        X: np.ndarray,
        F: np.ndarray,
        G: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        mode = self._deduplicate_in
        out_X, out_F, out_G = X, F, G

        if mode in {"objective", "both"}:
            keep_obj = _unique_rows_with_tolerance(out_F, self._objective_tolerance)
            out_X, out_F, out_G = _subset_arrays(out_X, out_F, out_G, keep_obj)

        if mode in {"decision", "both"}:
            keep_dec = _unique_rows_with_tolerance(np.asarray(out_X, dtype=float), self._decision_tolerance)
            out_X, out_F, out_G = _subset_arrays(out_X, out_F, out_G, keep_dec)

        return out_X, out_F, out_G

    def _replace_contents(
        self,
        X: np.ndarray,
        F: np.ndarray,
        G: np.ndarray | None,
    ) -> None:
        new_size = int(F.shape[0])
        self._ensure_storage(new_size)

        self._size = new_size
        if new_size:
            np.copyto(self._X[:new_size], X)
            np.copyto(self._F[:new_size], F)

        if G is None:
            self._G = None
            return

        self._ensure_constraint_storage(G.shape[1], new_size)
        assert self._G is not None
        if new_size:
            np.copyto(self._G[:new_size], G)

    def _ensure_storage(self, min_rows: int) -> None:
        if self._X.shape[0] >= min_rows:
            return
        new_rows = max(min_rows, int(self._X.shape[0] * 1.5) + 1)
        new_X = np.empty((new_rows, self._n_var), dtype=self._dtype)
        new_F = np.empty((new_rows, self._n_obj), dtype=float)
        if self._size:
            new_X[: self._size] = self._X[: self._size]
            new_F[: self._size] = self._F[: self._size]
        self._X = new_X
        self._F = new_F

        if self._G is not None:
            n_con = int(self._G.shape[1])
            new_G = np.empty((new_rows, n_con), dtype=float)
            if self._size:
                new_G[: self._size] = self._G[: self._size]
            self._G = new_G

    def _ensure_constraint_storage(self, n_con: int, min_rows: int) -> None:
        if n_con <= 0:
            raise ValueError("n_con must be positive.")
        if self._G is not None and self._G.shape[1] != n_con:
            raise ValueError("Constraint column count mismatch.")
        if self._G is None:
            rows = max(self._X.shape[0], min_rows)
            self._G = np.empty((rows, n_con), dtype=float)
            return
        if self._G.shape[0] >= min_rows:
            return
        new_rows = max(min_rows, int(self._G.shape[0] * 1.5) + 1)
        new_G = np.empty((new_rows, n_con), dtype=float)
        if self._size:
            new_G[: self._size] = self._G[: self._size]
        self._G = new_G

    def _select_subset(
        self,
        F: np.ndarray,
        target_size: int,
        G: np.ndarray | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class CrowdingDistanceArchive(_BaseArchive):
    """
    Bounded archive with NSGA-II-style iterative crowding truncation.
    """

    def _select_subset(
        self,
        F: np.ndarray,
        target_size: int,
        G: np.ndarray | None = None,
    ) -> np.ndarray:
        keep = np.arange(F.shape[0], dtype=int)
        while keep.size > target_size:
            crowd = _single_front_crowding(F[keep])
            worst_local = int(np.argmin(crowd))
            keep = np.delete(keep, worst_local)
        return keep


class HypervolumeArchive(_BaseArchive):
    """
    Bounded archive with iterative hypervolume-contribution truncation.
    """

    def __init__(
        self,
        capacity: int,
        n_var: int,
        n_obj: int,
        dtype: Any,
        *,
        ref_offset: float = 1.0,
        ref_point: np.ndarray | list[float] | None = None,
        truncate_size: int | None = None,
        objective_tolerance: float = 1e-10,
        deduplicate_in: DeduplicateIn = "objective",
        decision_tolerance: float = 1e-32,
        n_con: int | None = None,
    ) -> None:
        super().__init__(
            capacity,
            n_var,
            n_obj,
            dtype,
            truncate_size=truncate_size,
            objective_tolerance=objective_tolerance,
            deduplicate_in=deduplicate_in,
            decision_tolerance=decision_tolerance,
            n_con=n_con,
        )
        self._ref_offset = float(ref_offset)
        self._global_worst: np.ndarray | None = None
        self._fixed_ref: np.ndarray | None = None
        if ref_point is not None:
            ref = np.asarray(ref_point, dtype=float)
            if ref.ndim != 1 or ref.shape[0] != self._n_obj:
                raise ValueError(
                    f"hv_ref_point must be 1D with length {self._n_obj}, got shape {ref.shape}."
                )
            self._fixed_ref = ref.copy()

    def _stable_ref(self, F: np.ndarray) -> np.ndarray:
        current_max = np.max(F, axis=0)
        if self._global_worst is None:
            self._global_worst = current_max.copy()
        else:
            np.maximum(self._global_worst, current_max, out=self._global_worst)
        return self._global_worst + self._ref_offset

    def _reference(self, F: np.ndarray) -> np.ndarray:
        if self._fixed_ref is not None:
            return self._fixed_ref
        return self._stable_ref(F)

    def _select_subset(
        self,
        F: np.ndarray,
        target_size: int,
        G: np.ndarray | None = None,
    ) -> np.ndarray:
        keep = np.arange(F.shape[0], dtype=int)
        while keep.size > target_size:
            F_keep = F[keep]
            ref = self._reference(F_keep)
            contrib = _hv_contributions(F_keep, ref)
            worst_local = int(np.argmin(contrib))
            keep = np.delete(keep, worst_local)
        return keep


class SPEA2Archive(_BaseArchive):
    """
    Bounded archive with SPEA2-style truncation.

    Uses strength raw fitness for convergence ranking and distance-based
    truncation on the splitting front for diversity.
    """

    def __init__(
        self,
        capacity: int,
        n_var: int,
        n_obj: int,
        dtype: Any,
        *,
        truncate_size: int | None = None,
        objective_tolerance: float = 1e-10,
        deduplicate_in: DeduplicateIn = "objective",
        decision_tolerance: float = 1e-32,
        n_con: int | None = None,
        constraint_mode: str = "feasibility",
    ) -> None:
        super().__init__(
            capacity,
            n_var,
            n_obj,
            dtype,
            truncate_size=truncate_size,
            objective_tolerance=objective_tolerance,
            deduplicate_in=deduplicate_in,
            decision_tolerance=decision_tolerance,
            n_con=n_con,
        )
        self._constraint_mode = str(constraint_mode or "none")

    def _select_subset(
        self,
        F: np.ndarray,
        target_size: int,
        G: np.ndarray | None = None,
    ) -> np.ndarray:
        from vamos.engine.algorithm.spea2.helpers import (
            dominance_matrix,
            strength_raw_fitness,
            truncate_by_distance,
        )

        n = int(F.shape[0])
        if n <= target_size:
            return np.arange(n, dtype=int)

        mode = self._constraint_mode if G is not None else "none"
        dom, _, _ = dominance_matrix(F, G, mode)
        raw_fitness = strength_raw_fitness(dom)

        selected: list[int] = []
        unique_raw = np.sort(np.unique(raw_fitness))
        for fit in unique_raw:
            front = np.flatnonzero(raw_fitness == fit)
            if len(selected) + front.size <= target_size:
                selected.extend(front.tolist())
                continue

            remaining = target_size - len(selected)
            if remaining <= 0:
                break

            front_F = F[front]
            dist_matrix = np.linalg.norm(front_F[:, None, :] - front_F[None, :, :], axis=2)
            keep_local = truncate_by_distance(dist_matrix, remaining)
            selected.extend(front[keep_local].tolist())
            break

        return np.asarray(selected, dtype=int)


class UnboundedArchive:
    """
    Archive that keeps all non-dominated solutions without size limit.
    """

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        dtype: Any,
        *,
        objective_tolerance: float = 1e-10,
        deduplicate_in: DeduplicateIn = "objective",
        decision_tolerance: float = 1e-32,
        n_con: int | None = None,
        initial_capacity: int = 256,
    ) -> None:
        if objective_tolerance < 0.0:
            raise ValueError("objective_tolerance must be >= 0.")
        if decision_tolerance < 0.0:
            raise ValueError("decision_tolerance must be >= 0.")
        if deduplicate_in not in {"objective", "decision", "both"}:
            raise ValueError("deduplicate_in must be 'objective', 'decision', or 'both'.")
        if n_con is not None and int(n_con) <= 0:
            raise ValueError("n_con must be positive when provided.")

        self._dtype = np.dtype(dtype)
        self._n_var = int(n_var)
        self._n_obj = int(n_obj)
        self._objective_tolerance = float(objective_tolerance)
        self._decision_tolerance = float(decision_tolerance)
        self._deduplicate_in = deduplicate_in
        self._n_con = int(n_con) if n_con is not None else None
        self._size = 0

        capacity = max(1, int(initial_capacity))
        self._X = np.empty((capacity, self._n_var), dtype=self._dtype)
        self._F = np.empty((capacity, self._n_obj), dtype=float)
        self._G = np.empty((capacity, self._n_con), dtype=float) if self._n_con is not None else None

    def update(
        self,
        population_X: np.ndarray,
        population_F: np.ndarray,
        population_G: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pop_X = np.asarray(population_X, dtype=self._dtype, order="C")
        pop_F = np.asarray(population_F, dtype=float, order="C")
        if pop_X.ndim != 2:
            raise ValueError("population_X must be a 2D array.")
        if pop_F.ndim != 2:
            raise ValueError("population_F must be a 2D array.")
        if pop_X.shape[0] != pop_F.shape[0]:
            raise ValueError("population_X and population_F must have the same number of rows.")
        if pop_X.shape[1] != self._n_var:
            raise ValueError(f"population_X has {pop_X.shape[1]} columns, expected {self._n_var}.")
        if pop_F.shape[1] != self._n_obj:
            raise ValueError(f"population_F has {pop_F.shape[1]} columns, expected {self._n_obj}.")
        if pop_X.shape[0] == 0:
            return self._snapshot()

        pop_G: np.ndarray | None = None
        if population_G is not None:
            pop_G = np.asarray(population_G, dtype=float, order="C")
            if pop_G.ndim != 2:
                raise ValueError("population_G must be a 2D array when provided.")
            if pop_G.shape[0] != pop_X.shape[0]:
                raise ValueError("population_G must have the same number of rows as population_X.")

        X_comb, F_comb, G_comb = self._merge_with_existing(pop_X, pop_F, pop_G)
        X_work, F_work, G_work = _BaseArchive._apply_feasibility_filter(X_comb, F_comb, G_comb)
        if F_work.shape[0] == 0:
            self._replace_contents(X_work, F_work, G_work)
            return self._snapshot()

        nd_mask = _nondominated_mask(F_work)
        X_nd = X_work[nd_mask]
        F_nd = F_work[nd_mask]
        G_nd = G_work[nd_mask] if G_work is not None else None

        X_nd, F_nd, G_nd = self._dedupe(X_nd, F_nd, G_nd)
        self._replace_contents(X_nd, F_nd, G_nd)
        return self._snapshot()

    def contents(self) -> tuple[np.ndarray, np.ndarray]:
        return self._snapshot()

    def _snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        return self._X[: self._size].copy(), self._F[: self._size].copy()

    def _merge_with_existing(
        self,
        pop_X: np.ndarray,
        pop_F: np.ndarray,
        pop_G: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self._size == 0:
            return pop_X, pop_F, pop_G

        total = self._size + pop_X.shape[0]
        X_comb = np.empty((total, self._n_var), dtype=self._dtype)
        F_comb = np.empty((total, self._n_obj), dtype=float)
        X_comb[: self._size] = self._X[: self._size]
        F_comb[: self._size] = self._F[: self._size]
        X_comb[self._size :] = pop_X
        F_comb[self._size :] = pop_F

        if self._G is None and pop_G is None:
            return X_comb, F_comb, None

        if pop_G is not None:
            n_con = pop_G.shape[1]
        else:
            assert self._G is not None
            n_con = self._G.shape[1]
        if self._G is not None and self._G.shape[1] != n_con:
            raise ValueError("Constraint column count mismatch with existing archive contents.")
        if pop_G is not None and pop_G.shape[1] != n_con:
            raise ValueError("population_G has an incompatible number of constraint columns.")

        G_comb = np.zeros((total, n_con), dtype=float)
        if self._G is not None and self._size > 0:
            G_comb[: self._size] = self._G[: self._size]
        if pop_G is not None:
            G_comb[self._size :] = pop_G
        return X_comb, F_comb, G_comb

    def _dedupe(
        self,
        X: np.ndarray,
        F: np.ndarray,
        G: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        mode = self._deduplicate_in
        out_X, out_F, out_G = X, F, G

        if mode in {"objective", "both"}:
            keep_obj = _unique_rows_with_tolerance(out_F, self._objective_tolerance)
            out_X, out_F, out_G = _subset_arrays(out_X, out_F, out_G, keep_obj)

        if mode in {"decision", "both"}:
            keep_dec = _unique_rows_with_tolerance(np.asarray(out_X, dtype=float), self._decision_tolerance)
            out_X, out_F, out_G = _subset_arrays(out_X, out_F, out_G, keep_dec)

        return out_X, out_F, out_G

    def _replace_contents(
        self,
        X: np.ndarray,
        F: np.ndarray,
        G: np.ndarray | None,
    ) -> None:
        new_size = int(F.shape[0])
        self._ensure_storage(new_size)
        self._size = new_size
        if self._size:
            np.copyto(self._X[: self._size], X)
            np.copyto(self._F[: self._size], F)

        if G is None:
            self._G = None
            return

        self._ensure_constraint_storage(G.shape[1], new_size)
        assert self._G is not None
        if self._size:
            np.copyto(self._G[: self._size], G)

    def _ensure_storage(self, min_rows: int) -> None:
        if self._X.shape[0] >= min_rows:
            return
        new_rows = max(min_rows, int(self._X.shape[0] * 1.5) + 1)
        new_X = np.empty((new_rows, self._n_var), dtype=self._dtype)
        new_F = np.empty((new_rows, self._n_obj), dtype=float)
        if self._size:
            new_X[: self._size] = self._X[: self._size]
            new_F[: self._size] = self._F[: self._size]
        self._X = new_X
        self._F = new_F

        if self._G is not None:
            n_con = int(self._G.shape[1])
            new_G = np.empty((new_rows, n_con), dtype=float)
            if self._size:
                new_G[: self._size] = self._G[: self._size]
            self._G = new_G

    def _ensure_constraint_storage(self, n_con: int, min_rows: int) -> None:
        if n_con <= 0:
            raise ValueError("n_con must be positive.")
        if self._G is not None and self._G.shape[1] != n_con:
            raise ValueError("Constraint column count mismatch.")
        if self._G is None:
            rows = max(self._X.shape[0], min_rows)
            self._G = np.empty((rows, n_con), dtype=float)
            return
        if self._G.shape[0] >= min_rows:
            return
        new_rows = max(min_rows, int(self._G.shape[0] * 1.5) + 1)
        new_G = np.empty((new_rows, n_con), dtype=float)
        if self._size:
            new_G[: self._size] = self._G[: self._size]
        self._G = new_G


__all__ = [
    "HypervolumeArchive",
    "CrowdingDistanceArchive",
    "SPEA2Archive",
    "UnboundedArchive",
    "_single_front_crowding",
    "_hv_contributions",
]
