from __future__ import annotations

from typing import Any, cast

import numpy as np

try:  # pragma: no cover - optional dependency
    import moocore as _moocore
except ImportError:  # pragma: no cover - optional dependency
    _moocore = None

from vamos.foundation.kernel.numpy_backend import _compute_crowding


def _single_front_crowding(F: np.ndarray) -> np.ndarray:
    """Crowding distance for a single nondominated front."""
    if F.shape[0] == 0:
        return np.empty(0, dtype=float)
    fronts = [list(range(F.shape[0]))]
    return _compute_crowding(F, fronts)


def _hv_contributions(F: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Compute hypervolume contribution of each point.
    Uses moocore if available, otherwise falls back to crowding distance.
    """
    if F.shape[0] == 0:
        return np.empty(0, dtype=float)
    if _moocore is not None:
        return np.asarray(_moocore.hv_contributions(F, ref=ref), dtype=float)
    # Fallback: use crowding distance as proxy (not ideal but functional)
    return _single_front_crowding(F)


class _BaseArchive:
    """
    Base class for external archives that keep nondominated solutions.
    Subclasses implement different pruning strategies via _get_indicator().
    """

    def __init__(self, capacity: int, n_var: int, n_obj: int, dtype: Any, objective_tolerance: float = 1e-10) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("archive capacity must be positive.")
        self._dtype = np.dtype(dtype)
        self._n_var = int(n_var)
        self._n_obj = int(n_obj)
        self._objective_tolerance = float(objective_tolerance)
        storage_rows = self.capacity + 1  # allow temporary overflow before trimming
        self._X = np.empty((storage_rows, self._n_var), dtype=self._dtype)
        self._F = np.empty((storage_rows, self._n_obj), dtype=float)
        self._size = 0

    def update(
        self,
        population_X: np.ndarray,
        population_F: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pop_X = np.asarray(population_X, dtype=self._dtype, order="C")
        pop_F = np.asarray(population_F, dtype=float, order="C")
        for i in range(pop_X.shape[0]):
            x = pop_X[i]
            f = pop_F[i]
            if self._size == 0:
                self._append(x, f)
                continue
            if self._is_dominated(f):
                continue
            if self._is_duplicate(f):
                continue
            self._remove_dominated(f)
            self._append(x, f)
            if self._size > self.capacity:
                self._trim_to_capacity()
        return self._snapshot()

    def contents(self) -> tuple[np.ndarray, np.ndarray]:
        return self._snapshot()

    def _snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        return self._X[: self._size].copy(), self._F[: self._size].copy()

    def _append(self, x: np.ndarray, f: np.ndarray) -> None:
        self._ensure_storage(self._size + 1)
        np.copyto(self._X[self._size], x)
        np.copyto(self._F[self._size], f)
        self._size += 1

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

    def _is_dominated(self, f: np.ndarray) -> bool:
        existing = self._F[: self._size]
        return bool(self._dominates(existing, f).any()) if existing.shape[0] else False

    def _is_duplicate(self, f: np.ndarray) -> bool:
        if self._size == 0:
            return False
        existing = self._F[: self._size]
        diff = np.abs(existing - f)
        return bool(np.any(np.all(diff <= self._objective_tolerance, axis=1)))

    @staticmethod
    def _dominates(candidates: np.ndarray, f: np.ndarray) -> np.ndarray:
        if candidates.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        less_equal = candidates <= f
        strictly_less = candidates < f
        return cast(np.ndarray, np.all(less_equal, axis=1) & np.any(strictly_less, axis=1))

    def _remove_dominated(self, f: np.ndarray) -> None:
        """Remove existing solutions that are dominated by the new point f."""
        if self._size == 0:
            return
        existing = self._F[: self._size]
        f_leq_existing = f <= existing
        f_lt_existing = f < existing
        dominated_by_f = np.all(f_leq_existing, axis=1) & np.any(f_lt_existing, axis=1)
        if not dominated_by_f.any():
            return
        keep_mask = ~dominated_by_f
        self._compress_in_place(keep_mask)

    def _compress_in_place(self, keep_mask: np.ndarray) -> None:
        write = 0
        for read in range(self._size):
            if keep_mask[read]:
                if write != read:
                    self._X[write] = self._X[read]
                    self._F[write] = self._F[read]
                write += 1
        self._size = write

    def _get_indicator(self, F: np.ndarray) -> np.ndarray:
        """Return indicator values for each solution. Higher = better (keep)."""
        raise NotImplementedError

    def _trim_to_capacity(self) -> None:
        """Remove solution with smallest indicator value."""
        if self._size <= self.capacity:
            return
        active_F = self._F[: self._size]
        indicator = self._get_indicator(active_F)
        # Remove the solution with smallest indicator
        worst_idx = int(np.argmin(indicator))
        # Shift remaining solutions
        if worst_idx < self._size - 1:
            self._X[worst_idx : self._size - 1] = self._X[worst_idx + 1 : self._size]
            self._F[worst_idx : self._size - 1] = self._F[worst_idx + 1 : self._size]
        self._size -= 1


class CrowdingDistanceArchive(_BaseArchive):
    """
    External archive that keeps nondominated solutions and trims using
    crowding distance. When capacity is exceeded, the solution with the
    smallest crowding distance is removed (matching jMetal behavior).
    """

    def _get_indicator(self, F: np.ndarray) -> np.ndarray:
        return _single_front_crowding(F)


class HypervolumeArchive(_BaseArchive):
    """
    External archive that keeps nondominated solutions and trims using
    hypervolume contributions. When capacity is exceeded, the solution
    with the smallest hypervolume contribution is removed (SMS-EMOA style).

    The reference point is dynamically computed from the current archive.
    """

    def __init__(self, capacity: int, n_var: int, n_obj: int, dtype: Any, ref_offset: float = 1.0) -> None:
        super().__init__(capacity, n_var, n_obj, dtype)
        self._ref_offset = float(ref_offset)

    def _get_indicator(self, F: np.ndarray) -> np.ndarray:
        ref = np.max(F, axis=0) + self._ref_offset
        return _hv_contributions(F, ref)


class UnboundedArchive:
    """
    External archive that keeps all non-dominated solutions without a size limit.

    This is useful when you want the final archive to reflect the union of all
    non-dominated solutions encountered during the run. Unlike the bounded
    archives above, no trimming is performed.
    """

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        dtype: Any,
        *,
        objective_tolerance: float = 1e-10,
        initial_capacity: int = 256,
    ) -> None:
        self._dtype = np.dtype(dtype)
        self._n_var = int(n_var)
        self._n_obj = int(n_obj)
        self._objective_tolerance = float(objective_tolerance)
        self._size = 0

        capacity = max(1, int(initial_capacity))
        self._X = np.empty((capacity, self._n_var), dtype=self._dtype)
        self._F = np.empty((capacity, self._n_obj), dtype=float)

    def update(
        self,
        population_X: np.ndarray,
        population_F: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pop_X = np.asarray(population_X, dtype=self._dtype, order="C")
        pop_F = np.asarray(population_F, dtype=float, order="C")
        if pop_X.size == 0:
            return self._snapshot()

        for i in range(pop_X.shape[0]):
            x = pop_X[i]
            f = pop_F[i]
            if self._size == 0:
                self._append(x, f)
                continue
            if self._is_dominated(f):
                continue
            if self._is_duplicate(f):
                continue
            self._remove_dominated(f)
            self._append(x, f)

        return self._snapshot()

    def contents(self) -> tuple[np.ndarray, np.ndarray]:
        return self._snapshot()

    def _snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        return self._X[: self._size], self._F[: self._size]

    def _append(self, x: np.ndarray, f: np.ndarray) -> None:
        self._ensure_storage(self._size + 1)
        np.copyto(self._X[self._size], x)
        np.copyto(self._F[self._size], f)
        self._size += 1

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

    def _is_dominated(self, f: np.ndarray) -> bool:
        if self._size == 0:
            return False
        existing = self._F[: self._size]
        return bool(_BaseArchive._dominates(existing, f).any())

    def _is_duplicate(self, f: np.ndarray) -> bool:
        if self._size == 0:
            return False
        existing = self._F[: self._size]
        diff = np.abs(existing - f)
        return bool(np.any(np.all(diff <= self._objective_tolerance, axis=1)))

    def _remove_dominated(self, f: np.ndarray) -> None:
        if self._size == 0:
            return
        existing = self._F[: self._size]
        f_leq_existing = f <= existing
        f_lt_existing = f < existing
        dominated_by_f = np.all(f_leq_existing, axis=1) & np.any(f_lt_existing, axis=1)
        if not dominated_by_f.any():
            return

        keep_mask = ~dominated_by_f
        write = 0
        for read in range(self._size):
            if keep_mask[read]:
                if write != read:
                    self._X[write] = self._X[read]
                    self._F[write] = self._F[read]
                write += 1
        self._size = write


__all__ = [
    "HypervolumeArchive",
    "CrowdingDistanceArchive",
    "UnboundedArchive",
    "_single_front_crowding",
    "_hv_contributions",
]
