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
    crowding distance. The archive owns contiguous buffers and only copies
    data when its contents change.
    """

    def __init__(self, capacity: int, n_var: int, n_obj: int, dtype):
        self.capacity = int(capacity)
        self._dtype = np.dtype(dtype)
        self._X = np.empty((0, n_var), dtype=self._dtype)
        self._F = np.empty((0, n_obj), dtype=float)

    def update(self, population_X: np.ndarray, population_F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pop_X = np.asarray(population_X, dtype=self._dtype, order="C")
        pop_F = np.asarray(population_F, dtype=float, order="C")

        if self._X.size == 0:
            current_X = pop_X.copy()
            current_F = pop_F.copy()
        else:
            current_X = np.vstack((self._X, pop_X))
            current_F = np.vstack((self._F, pop_F))

        if _moocore is not None:
            mask = _moocore.is_nondominated(current_F, keep_weakly=False)
        else:
            fronts, _ = _fast_non_dominated_sort(current_F)
            mask = np.zeros(current_F.shape[0], dtype=bool)
            if fronts:
                mask[np.asarray(fronts[0], dtype=int)] = True

        nd_X = current_X[mask]
        nd_F = current_F[mask]
        if nd_X.shape[0] <= self.capacity:
            self._X = nd_X
            self._F = nd_F
            return self._X, self._F

        if _moocore is not None:
            ref = np.max(nd_F, axis=0) + 1.0
            indicator = np.asarray(_moocore.hv_contributions(nd_F, ref=ref), dtype=float)
        else:
            indicator = _single_front_crowding(nd_F)
        order = np.argsort(indicator)[::-1][: self.capacity]
        self._X = nd_X[order]
        self._F = nd_F[order]
        return self._X, self._F

    def contents(self) -> tuple[np.ndarray, np.ndarray]:
        return self._X.copy(), self._F.copy()

