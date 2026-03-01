from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal, overload

import numpy as np

from .backend import KernelBackend
from .native_bridge import NativeNsga2Bridge
from .numba_backend import NumbaKernel, _compute_crowding_numba


class NumbaMixedKernel(KernelBackend):
    """Explicit opt-in kernel that mixes Numba orchestration with safe native kernels."""

    name = "numba-mixed"

    def __init__(self) -> None:
        self._numba = NumbaKernel()
        self._native = NativeNsga2Bridge()
        self._native.require_native()
        self._native_survival_enabled = False
        self._native_calls: list[str] = []

    @property
    def used_native_for(self) -> list[str]:
        return list(self._native_calls)

    def _mark(self, method: str) -> None:
        self._native_calls.append(method)

    def capabilities(self) -> Iterable[str]:
        return ("cpu", "native", "native:nsga2", "native:rank2d", "numba")

    def quality_indicators(self) -> Iterable[str]:
        return self._numba.quality_indicators()

    def _can_use_native_ranking(self, F: np.ndarray) -> bool:
        return F.ndim == 2 and F.shape[1] == 2

    def nsga2_ranking(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        F_arr = np.asarray(F)
        if not self._can_use_native_ranking(F_arr):
            return self._numba.nsga2_ranking(F_arr)

        F_native = self._native._as_float64_c(F_arr, name="F", ndim=2)
        _fronts, ranks = self._native.fast_non_dominated_sort(F_native)
        crowding = _compute_crowding_numba(F_native, ranks)
        self._mark("nsga2_ranking")
        return np.ascontiguousarray(ranks, dtype=np.int64), np.ascontiguousarray(crowding, dtype=np.float64)

    def tournament_selection(
        self,
        ranks: np.ndarray,
        crowding: np.ndarray,
        pressure: int,
        rng: np.random.Generator,
        n_parents: int,
    ) -> np.ndarray:
        return self._numba.tournament_selection(ranks, crowding, pressure, rng, n_parents)

    def sbx_crossover(
        self,
        X_parents: np.ndarray,
        params: Mapping[str, object],
        rng: np.random.Generator | None,
        xl: float | np.ndarray,
        xu: float | np.ndarray,
    ) -> np.ndarray:
        return self._numba.sbx_crossover(X_parents, params, rng, xl, xu)

    def polynomial_mutation(
        self,
        X: np.ndarray,
        params: Mapping[str, object],
        rng: np.random.Generator | None,
        xl: float | np.ndarray,
        xu: float | np.ndarray,
    ) -> None:
        self._numba.polynomial_mutation(X, params, rng, xl, xu)

    @overload
    def nsga2_survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
        return_indices: Literal[False] = False,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def nsga2_survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
        return_indices: Literal[True],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def nsga2_survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
        return_indices: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._native_survival_enabled:
            self._mark("nsga2_survival")
            return self._native.nsga2_survival(X, F, X_off, F_off, pop_size, return_indices=return_indices)
        return self._numba.nsga2_survival(X, F, X_off, F_off, pop_size, return_indices=return_indices)

    def hypervolume(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        return self._numba.hypervolume(points, reference_point)


__all__ = ["NumbaMixedKernel"]
