from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    import moocore as _moocore  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _moocore = None

from vamos.kernel.numpy_backend import _fast_non_dominated_sort, _compute_crowding


def _single_front_crowding(F: np.ndarray) -> np.ndarray:
    """Crowding distance for a single nondominated front."""
    if F.shape[0] == 0:
        return np.empty(0, dtype=float)
    fronts = [list(range(F.shape[0]))]
    return _compute_crowding(F, fronts)


class CrowdingDistanceArchive:
    """
    External archive that keeps nondominated solutions and trims them using
    either hypervolume contributions (when MooCore is available) or classical
    crowding distance. The archive stores solutions inside fixed buffers and
    performs incremental updates to avoid repeated full re-sorts.
    """

    def __init__(self, capacity: int, n_var: int, n_obj: int, dtype):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("archive capacity must be positive.")
        self._dtype = np.dtype(dtype)
        self._n_var = int(n_var)
        self._n_obj = int(n_obj)
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
        return self._dominates(existing, f).any() if existing.shape[0] else False

    @staticmethod
    def _dominates(candidates: np.ndarray, f: np.ndarray) -> np.ndarray:
        if candidates.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        less_equal = candidates <= f
        strictly_less = candidates < f
        return np.all(less_equal, axis=1) & np.any(strictly_less, axis=1)

    def _remove_dominated(self, f: np.ndarray) -> None:
        if self._size == 0:
            return
        dominated = self._dominates(self._F[: self._size], f)
        if not dominated.any():
            return
        keep_mask = ~dominated
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

    def _trim_to_capacity(self) -> None:
        if self._size <= self.capacity:
            return
        active_F = self._F[: self._size]
        if _moocore is not None and active_F.size:
            ref = np.max(active_F, axis=0) + 1.0
            indicator = np.asarray(_moocore.hv_contributions(active_F, ref=ref), dtype=float)
        else:
            indicator = _single_front_crowding(active_F)
        order = np.argsort(indicator)[::-1][: self.capacity]
        active_X = self._X[: self._size]
        active_F = self._F[: self._size]
        self._X[: self.capacity] = active_X[order]
        self._F[: self.capacity] = active_F[order]
        self._size = self.capacity
