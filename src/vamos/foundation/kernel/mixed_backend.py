from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal, overload

import numpy as np

from .backend import KernelBackend
from .cpp_backend import CppBackend
from .native_bridge import NativeNsga2Bridge
from .numba_backend import NumbaKernel, _compute_crowding_numba, _select_nsga2_indices


class NumbaMixedKernel(KernelBackend):
    """Explicit opt-in kernel that mixes Numba orchestration with safe native kernels."""

    name = "vamos-numba"

    def __init__(self) -> None:
        self._numba = NumbaKernel()
        self._native = NativeNsga2Bridge()
        self._native.require_native()
        self._cpp = CppBackend()
        self._native_survival_enabled = True
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

    def _can_use_native_survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
    ) -> bool:
        if not self._native_survival_enabled:
            return False
        if X.ndim != 2 or F.ndim != 2 or X_off.ndim != 2 or F_off.ndim != 2:
            return False
        if X.shape[0] != F.shape[0] or X_off.shape[0] != F_off.shape[0]:
            return False
        if X.shape[1] != X_off.shape[1] or F.shape[1] != F_off.shape[1]:
            return False
        return F.shape[1] == 2

    def _native_ranking(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        F_native = self._native._as_float64_c(F, name="F", ndim=2)
        fronts, ranks = self._native.fast_non_dominated_sort(F_native)
        try:
            crowding = self._native.crowding_distance(F_native, fronts)
        except Exception:
            crowding = _compute_crowding_numba(F_native, ranks)
        return np.ascontiguousarray(ranks, dtype=np.int64), np.ascontiguousarray(crowding, dtype=np.float64)

    def nsga2_ranking(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        F_arr = np.asarray(F)
        if not self._can_use_native_ranking(F_arr):
            return self._numba.nsga2_ranking(F_arr)

        ranks, crowding = self._native_ranking(F_arr)
        self._mark("nsga2_ranking")
        return ranks, crowding

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

    def generate_offspring(
        self,
        X: np.ndarray,
        F: np.ndarray,
        params: Mapping[str, object],
        rng: np.random.Generator,
        xl: np.ndarray | float,
        xu: np.ndarray | float,
        n_offspring: int | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        F_arr = np.ascontiguousarray(np.asarray(F, dtype=np.float64))
        if np.isscalar(xl):
            xl_arr = np.full(X_arr.shape[1], float(xl), dtype=np.float64)
        else:
            xl_arr = np.ascontiguousarray(np.asarray(xl, dtype=np.float64))
        if np.isscalar(xu):
            xu_arr = np.full(X_arr.shape[1], float(xu), dtype=np.float64)
        else:
            xu_arr = np.ascontiguousarray(np.asarray(xu, dtype=np.float64))
        try:
            offspring = self._cpp.generate_offspring(
                X_arr,
                F_arr,
                params,
                rng,
                xl_arr,
                xu_arr,
                n_offspring=n_offspring,
                out=out,
            )
        except Exception:
            ranks, crowding = self.nsga2_ranking(F_arr)
            pressure = int(params.get("selection_pressure", 2))
            parent_count = int(n_offspring) if n_offspring is not None else int(X_arr.shape[0])
            parent_count = max(2, parent_count)
            parent_idx = self.tournament_selection(ranks, crowding, pressure, rng, n_parents=parent_count)
            offspring = self.sbx_crossover(X_arr[parent_idx], params.get("crossover", {}), rng, xl_arr, xu_arr)
            self.polynomial_mutation(offspring, params.get("mutation", {}), rng, xl_arr, xu_arr)
            if n_offspring is not None and offspring.shape[0] > int(n_offspring):
                return offspring[: int(n_offspring)]
            return offspring
        self._mark("generate_offspring")
        return np.ascontiguousarray(np.asarray(offspring, dtype=np.float64))

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
        X_arr = np.asarray(X)
        F_arr = np.asarray(F)
        X_off_arr = np.asarray(X_off)
        F_off_arr = np.asarray(F_off)
        if self._can_use_native_survival(X_arr, F_arr, X_off_arr, F_off_arr):
            X_comb = self._numba._combine_rows(X_arr, X_off_arr, attr_name="_x_comb_buffer")
            F_comb = self._numba._combine_rows(F_arr, F_off_arr, attr_name="_f_comb_buffer")
            ranks, crowding = self._native_ranking(F_comb)
            sel = _select_nsga2_indices(ranks, crowding, pop_size)
            self._mark("nsga2_survival")
            if return_indices:
                return X_comb[sel], F_comb[sel], sel
            return X_comb[sel], F_comb[sel]
        return self._numba.nsga2_survival(
            X_arr,
            F_arr,
            X_off_arr,
            F_off_arr,
            pop_size,
            return_indices=return_indices,
        )

    def hypervolume(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        return self._numba.hypervolume(points, reference_point)


__all__ = ["NumbaMixedKernel"]
