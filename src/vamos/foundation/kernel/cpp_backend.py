from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Any, Literal, overload

import numpy as np

from .numpy_backend import NumPyKernel


def _as_float(value: object | None, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return float(default)


def _as_int(value: object | None, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, bool):
        return int(default)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return int(default)
    return int(default)


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


class CppBackend(NumPyKernel):
    """
    Native backend wrapper around the optional ``vamospp`` package.

    Behavior:
    - Uses C++ routines when the corresponding function exists in ``vamospp``.
    - Falls back to NumPyKernel implementations otherwise.
    """

    name = "cpp"

    def __init__(self) -> None:
        super().__init__()
        try:
            import vamospp
        except ImportError as exc:
            raise ImportError(
                "Kernel 'cpp' requires vamospp. Install with `pip install -e \".[native]\"` or `pip install vamospp`."
            ) from exc
        self._cpp = vamospp
        self._cpp_calls: list[str] = []

    @property
    def used_cpp_for(self) -> list[str]:
        return list(self._cpp_calls)

    def _mark(self, method: str) -> None:
        self._cpp_calls.append(method)

    def _seed(self, rng: np.random.Generator) -> int:
        return int(rng.integers(2**63))

    @staticmethod
    def _require_float64_c(value: np.ndarray, *, name: str, ndim: int) -> np.ndarray:
        arr = np.asarray(value)
        if arr.dtype != np.float64:
            raise TypeError(f"{name} must have dtype float64 for cpp backend.")
        if arr.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}D for cpp backend.")
        if not arr.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous for cpp backend.")
        return arr

    @staticmethod
    def _require_int64_c(value: np.ndarray, *, name: str, ndim: int) -> np.ndarray:
        arr = np.asarray(value)
        if arr.dtype != np.int64:
            raise TypeError(f"{name} must have dtype int64 for cpp backend.")
        if arr.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}D for cpp backend.")
        if not arr.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous for cpp backend.")
        return arr

    def _cpp_fn(self, name: str) -> Any | None:
        fn = getattr(self._cpp, name, None)
        return fn if callable(fn) else None

    def capabilities(self) -> Iterable[str]:
        tags = {"cpu", "cpp"}
        is_native = getattr(self._cpp, "is_native_backend", None)
        if callable(is_native) and bool(is_native()):
            tags.add("native")
        return tuple(sorted(tags))

    def quality_indicators(self) -> Iterable[str]:
        return ("hypervolume",)

    def nsga2_ranking(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fn = self._cpp_fn("nsga2_ranking")
        if fn is None:
            return super().nsga2_ranking(F)
        F_arr = self._require_float64_c(F, name="F", ndim=2)
        ranks, crowd = fn(F_arr)
        self._mark("nsga2_ranking")
        return np.asarray(ranks, dtype=np.int64), np.asarray(crowd, dtype=np.float64)

    def tournament_selection(
        self,
        ranks: np.ndarray,
        crowding: np.ndarray,
        pressure: int,
        rng: np.random.Generator,
        n_parents: int,
    ) -> np.ndarray:
        fn = self._cpp_fn("tournament_selection")
        if fn is None:
            return super().tournament_selection(ranks, crowding, pressure, rng, n_parents)
        ranks_arr = self._require_int64_c(ranks, name="ranks", ndim=1)
        crowd_arr = self._require_float64_c(crowding, name="crowding", ndim=1)
        winners = fn(ranks_arr, crowd_arr, int(pressure), self._seed(rng), int(n_parents))
        self._mark("tournament_selection")
        return np.asarray(winners, dtype=np.int64)

    def sbx_crossover(
        self,
        X_parents: np.ndarray,
        params: Mapping[str, object],
        rng: np.random.Generator,
        xl: float | np.ndarray,
        xu: float | np.ndarray,
    ) -> np.ndarray:
        fn = self._cpp_fn("sbx_crossover")
        if fn is None:
            return super().sbx_crossover(X_parents, params, rng, xl, xu)
        X_arr = self._require_float64_c(X_parents, name="X_parents", ndim=2)
        n_var = X_arr.shape[1]
        lower, upper = self._normalize_bounds(xl, xu, n_var)
        out = fn(
            X_arr,
            _as_float(params.get("prob"), 0.9),
            _as_float(params.get("eta"), 20.0),
            np.ascontiguousarray(lower, dtype=np.float64),
            np.ascontiguousarray(upper, dtype=np.float64),
            self._seed(rng),
        )
        self._mark("sbx_crossover")
        return self._require_float64_c(out, name="sbx_crossover output", ndim=2)

    def polynomial_mutation(
        self,
        X: np.ndarray,
        params: Mapping[str, object],
        rng: np.random.Generator,
        xl: float | np.ndarray,
        xu: float | np.ndarray,
    ) -> None:
        fn = self._cpp_fn("polynomial_mutation")
        if fn is None:
            super().polynomial_mutation(X, params, rng, xl, xu)
            return
        X_arr = self._require_float64_c(X, name="X", ndim=2)
        n_var = X_arr.shape[1]
        lower, upper = self._normalize_bounds(xl, xu, n_var)
        mutated = fn(
            X_arr,
            _as_float(params.get("prob"), 0.1),
            _as_float(params.get("eta"), 20.0),
            np.ascontiguousarray(lower, dtype=np.float64),
            np.ascontiguousarray(upper, dtype=np.float64),
            self._seed(rng),
            True,
        )
        X_arr[:] = self._require_float64_c(mutated, name="polynomial_mutation output", ndim=2)
        self._mark("polynomial_mutation")

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
        fn = self._cpp_fn("nsga2_survival")
        if fn is None:
            return super().nsga2_survival(X, F, X_off, F_off, pop_size, return_indices=return_indices)
        X_arr = self._require_float64_c(X, name="X", ndim=2)
        F_arr = self._require_float64_c(F, name="F", ndim=2)
        X_off_arr = self._require_float64_c(X_off, name="X_off", ndim=2)
        F_off_arr = self._require_float64_c(F_off, name="F_off", ndim=2)
        payload = fn(X_arr, F_arr, X_off_arr, F_off_arr, int(pop_size), bool(return_indices))
        self._mark("nsga2_survival")
        if return_indices:
            x_new, f_new, indices = payload
            return (
                self._require_float64_c(x_new, name="nsga2_survival X", ndim=2),
                self._require_float64_c(f_new, name="nsga2_survival F", ndim=2),
                self._require_int64_c(indices, name="nsga2_survival indices", ndim=1),
            )
        x_new, f_new = payload
        return (
            self._require_float64_c(x_new, name="nsga2_survival X", ndim=2),
            self._require_float64_c(f_new, name="nsga2_survival F", ndim=2),
        )

    def hypervolume(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        fn = self._cpp_fn("hypervolume")
        if fn is None:
            return super().hypervolume(points, reference_point)
        points_arr = self._require_float64_c(points, name="points", ndim=2)
        ref_arr = self._require_float64_c(reference_point, name="reference_point", ndim=1)
        self._mark("hypervolume")
        return float(fn(points_arr, ref_arr))

    def hypervolume_contributions(self, points: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
        fn = self._cpp_fn("hypervolume_contributions")
        points_arr = self._require_float64_c(points, name="points", ndim=2)
        ref_arr = self._require_float64_c(reference_point, name="reference_point", ndim=1)
        # Prefer the Python indicator layer for >=3D so MooCore (if installed)
        # can provide fast contributions; native fallback is currently expensive.
        if points_arr.shape[1] >= 3:
            from vamos.foundation.quality_indicators.hypervolume import hypervolume_contributions

            return np.asarray(hypervolume_contributions(points_arr, ref_arr), dtype=np.float64)
        if fn is None:
            from vamos.foundation.quality_indicators.hypervolume import hypervolume_contributions

            return np.asarray(hypervolume_contributions(points_arr, ref_arr), dtype=np.float64)
        self._mark("hypervolume_contributions")
        return np.asarray(fn(points_arr, ref_arr), dtype=np.float64)

    def smsemoa_remove_index(self, F_combined: np.ndarray, ref_point: np.ndarray) -> int:
        fn = self._cpp_fn("smsemoa_remove_index")
        F_arr = self._require_float64_c(F_combined, name="F_combined", ndim=2)
        ref_arr = self._require_float64_c(ref_point, name="ref_point", ndim=1)
        # Native 2D path is fast; >=3D falls back to indicator layer for speed.
        if fn is None or F_arr.shape[1] >= 3:
            ranks, _ = self.nsga2_ranking(F_combined)
            worst_rank = int(ranks.max(initial=0))
            worst_idx = np.flatnonzero(ranks == worst_rank)
            if worst_idx.size == 1:
                return int(worst_idx[0])
            contribs = self.hypervolume_contributions(F_combined[worst_idx], ref_point)
            return int(worst_idx[int(np.argmin(contribs))])
        self._mark("smsemoa_remove_index")
        return int(fn(F_arr, ref_arr))

    def dominance_matrix(self, F: np.ndarray) -> np.ndarray:
        fn = self._cpp_fn("dominance_matrix")
        if fn is None:
            less_equal = F[:, None, :] <= F[None, :, :]
            strictly_less = F[:, None, :] < F[None, :, :]
            dom = np.logical_and(np.all(less_equal, axis=2), np.any(strictly_less, axis=2))
            np.fill_diagonal(dom, False)
            return dom
        F_arr = self._require_float64_c(F, name="F", ndim=2)
        self._mark("dominance_matrix")
        return np.asarray(fn(F_arr), dtype=bool)

    def spea2_fitness(
        self,
        F: np.ndarray,
        dom: np.ndarray | None = None,
        k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        fn = self._cpp_fn("spea2_fitness")
        if fn is None:
            from vamos.engine.algorithm.spea2.helpers import spea2_fitness as _spea2_fitness

            dom_matrix = np.asarray(dom, dtype=bool) if dom is not None else self.dominance_matrix(F)
            return _spea2_fitness(F, dom_matrix, k)
        F_arr = self._require_float64_c(F, name="F", ndim=2)
        dom_arr = np.asarray(dom, dtype=bool) if dom is not None else self.dominance_matrix(F_arr)
        self._mark("spea2_fitness")
        fitness, dist = fn(F_arr, dom_arr, None if k is None else int(k))
        return (
            self._require_float64_c(fitness, name="spea2_fitness fitness", ndim=1),
            self._require_float64_c(dist, name="spea2_fitness dist", ndim=2),
        )

    def spea2_environmental_selection_indices(
        self,
        F: np.ndarray,
        archive_size: int,
        k: int | None = None,
    ) -> np.ndarray:
        fn = self._cpp_fn("spea2_environmental_selection_indices")
        F_arr = self._require_float64_c(F, name="F", ndim=2)
        keep = int(archive_size)
        if keep <= 0:
            return np.empty(0, dtype=np.int64)
        if F_arr.shape[0] <= keep:
            return np.arange(F_arr.shape[0], dtype=np.int64)

        if fn is not None:
            indices = fn(F_arr, keep, None if k is None else int(k))
            self._mark("spea2_environmental_selection_indices")
            return np.asarray(indices, dtype=np.int64)

        dom = self.dominance_matrix(F_arr)
        strength = dom.sum(axis=1).astype(np.float64)
        raw = np.zeros(F_arr.shape[0], dtype=np.float64)
        for i in range(F_arr.shape[0]):
            dominators = np.where(dom[:, i])[0]
            raw[i] = strength[dominators].sum()

        unique_fitness = np.unique(raw)
        selected: list[int] = []
        k_eff = int(k) if k is not None else 1
        if k_eff < 1:
            k_eff = 1

        for fit in np.sort(unique_fitness):
            front = np.flatnonzero(raw == fit)
            if len(selected) + front.size <= keep:
                selected.extend(front.tolist())
                continue

            remaining = keep - len(selected)
            if remaining <= 0:
                break
            from vamos.engine.algorithm.spea2.helpers import truncate_by_distance

            front_F = F_arr[front]
            dist_front = np.linalg.norm(front_F[:, None, :] - front_F[None, :, :], axis=2)
            local = truncate_by_distance(dist_front, remaining, k=k_eff)
            selected.extend(front[local].tolist())
            break

        return np.asarray(selected, dtype=np.int64)

    def ibea_indicator_matrix(
        self,
        F: np.ndarray,
        reference_point: np.ndarray | None = None,
        kind: str = "epsilon",
    ) -> np.ndarray:
        fn = self._cpp_fn("ibea_indicator_matrix")
        if fn is None:
            from vamos.engine.algorithm.ibea.helpers import compute_indicator_matrix

            return np.asarray(compute_indicator_matrix(F, kind), dtype=np.float64)
        F_arr = self._require_float64_c(F, name="F", ndim=2)
        if reference_point is None:
            ref = np.max(F_arr, axis=0) + 1.0
        else:
            ref = self._require_float64_c(reference_point, name="reference_point", ndim=1)
        ref_arr = self._require_float64_c(ref, name="reference_point", ndim=1)
        self._mark("ibea_indicator_matrix")
        return self._require_float64_c(fn(F_arr, ref_arr, str(kind)), name="ibea_indicator_matrix", ndim=2)

    def ibea_environmental_selection_indices(
        self,
        F: np.ndarray,
        pop_size: int,
        reference_point: np.ndarray | None = None,
        kind: str = "epsilon",
        kappa: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        fn = self._cpp_fn("ibea_environmental_selection_indices")
        F_arr = self._require_float64_c(F, name="F", ndim=2)
        keep = int(pop_size)
        if keep <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        if fn is not None:
            selected, fitness = fn(
                F_arr,
                keep,
                None if reference_point is None else self._require_float64_c(reference_point, name="reference_point", ndim=1),
                str(kind),
                float(kappa),
            )
            self._mark("ibea_environmental_selection_indices")
            return np.asarray(selected, dtype=np.int64), np.asarray(fitness, dtype=np.float64)

        indicator = self.ibea_indicator_matrix(F_arr, reference_point, kind)
        mat = np.asarray(indicator, dtype=np.float64).copy()
        np.fill_diagonal(mat, np.inf)
        denom = float(kappa) if float(kappa) != 0.0 else 1.0e-12
        contrib = np.exp(-mat / denom)
        contrib[~np.isfinite(contrib)] = 0.0
        fitness = np.asarray(-np.sum(contrib, axis=1), dtype=np.float64)
        selected = np.arange(F_arr.shape[0], dtype=np.int64)
        target = min(keep, F_arr.shape[0])

        while selected.size > target:
            worst = int(np.argmin(fitness))
            delta = np.exp(-indicator[:, int(selected[worst])] / denom)
            delta[~np.isfinite(delta)] = 0.0
            fitness += delta[selected]
            fitness[worst] -= delta[int(selected[worst])]
            selected = np.delete(selected, worst)
            fitness = np.delete(fitness, worst)
        return selected, np.asarray(fitness, dtype=np.float64)

    def smsemoa_generate_offspring(
        self,
        X: np.ndarray,
        F: np.ndarray,
        selection: str,
        pressure: int,
        params: Mapping[str, object],
        rng: np.random.Generator,
        xl: np.ndarray,
        xu: np.ndarray,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        fn = self._cpp_fn("smsemoa_generate_offspring")
        X_arr = self._require_float64_c(X, name="X", ndim=2)
        F_arr = self._require_float64_c(F, name="F", ndim=2)
        xl_arr = self._require_float64_c(xl, name="xl", ndim=1)
        xu_arr = self._require_float64_c(xu, name="xu", ndim=1)
        config = {
            "sbx_prob": _as_float(params.get("sbx_prob"), 0.9),
            "sbx_eta": _as_float(params.get("sbx_eta"), 20.0),
            "pm_prob": _as_float(params.get("pm_prob"), 1.0 / max(1, X_arr.shape[1])),
            "pm_eta": _as_float(params.get("pm_eta"), 20.0),
        }
        sel_name = str(selection).strip().lower()
        if fn is not None:
            child = fn(X_arr, F_arr, sel_name, int(pressure), xl_arr, xu_arr, config, self._seed(rng), out)
            self._mark("smsemoa_generate_offspring")
            return self._require_float64_c(child, name="smsemoa_generate_offspring output", ndim=2)

        if sel_name == "tournament":
            if X_arr.shape[0] > 1:
                ranks, crowd = self.nsga2_ranking(F_arr)
                parent_idx = self.tournament_selection(ranks, crowd, int(pressure), rng, n_parents=2)
            else:
                parent_idx = np.zeros(2, dtype=np.int64)
        else:
            parent_idx = rng.choice(X_arr.shape[0], size=2, replace=True).astype(np.int64)
        parents = X_arr[parent_idx]
        offspring = self.sbx_crossover(parents, {"prob": config["sbx_prob"], "eta": config["sbx_eta"]}, rng, xl_arr, xu_arr)
        child = np.asarray(offspring[:1], dtype=np.float64)
        self.polynomial_mutation(child, {"prob": config["pm_prob"], "eta": config["pm_eta"]}, rng, xl_arr, xu_arr)
        if out is not None:
            out_arr = self._require_float64_c(out, name="out", ndim=2)
            if out_arr.shape != child.shape:
                raise ValueError("out has wrong shape for smsemoa_generate_offspring.")
            out_arr[:] = child
            return out_arr
        return child

    def spea2_generate_offspring(
        self,
        X: np.ndarray,
        F: np.ndarray,
        n_offspring: int,
        k_neighbors: int,
        params: Mapping[str, object],
        rng: np.random.Generator,
        xl: np.ndarray,
        xu: np.ndarray,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        fn = self._cpp_fn("spea2_generate_offspring")
        X_arr = self._require_float64_c(X, name="X", ndim=2)
        F_arr = self._require_float64_c(F, name="F", ndim=2)
        xl_arr = self._require_float64_c(xl, name="xl", ndim=1)
        xu_arr = self._require_float64_c(xu, name="xu", ndim=1)
        target = int(n_offspring)
        if target <= 0:
            return np.empty((0, X_arr.shape[1]), dtype=np.float64)

        config = {
            "sbx_prob": _as_float(params.get("sbx_prob"), 0.9),
            "sbx_eta": _as_float(params.get("sbx_eta"), 20.0),
            "pm_prob": _as_float(params.get("pm_prob"), 1.0 / max(1, X_arr.shape[1])),
            "pm_eta": _as_float(params.get("pm_eta"), 20.0),
        }

        if fn is not None:
            offspring = fn(X_arr, F_arr, target, int(k_neighbors), xl_arr, xu_arr, config, self._seed(rng), out)
            self._mark("spea2_generate_offspring")
            return self._require_float64_c(offspring, name="spea2_generate_offspring output", ndim=2)

        dom = self.dominance_matrix(F_arr)
        strength = dom.sum(axis=1).astype(np.float64)
        raw_fitness = np.zeros(F_arr.shape[0], dtype=np.float64)
        for i in range(F_arr.shape[0]):
            dominators = np.where(dom[:, i])[0]
            raw_fitness[i] = strength[dominators].sum()
        dist = np.linalg.norm(F_arr[:, None, :] - F_arr[None, :, :], axis=2)
        k_eff = max(1, int(k_neighbors))
        if F_arr.shape[0] > 1:
            k_eff = min(k_eff, F_arr.shape[0] - 1)
        ranks = np.argsort(np.argsort(raw_fitness, kind="mergesort"), kind="mergesort").astype(np.int64)
        density = np.partition(dist, kth=k_eff, axis=1)[:, k_eff].astype(np.float64)
        if F_arr.shape[0] > 1:
            parent_idx = self.tournament_selection(ranks, density, 2, rng, n_parents=target * 2)
        else:
            parent_idx = np.zeros(target * 2, dtype=np.int64)
        parents = X_arr[parent_idx]
        crossed = self.sbx_crossover(parents, {"prob": config["sbx_prob"], "eta": config["sbx_eta"]}, rng, xl_arr, xu_arr)
        offspring = crossed.reshape(target, 2, X_arr.shape[1])[:, 0, :].copy()
        self.polynomial_mutation(offspring, {"prob": config["pm_prob"], "eta": config["pm_eta"]}, rng, xl_arr, xu_arr)
        if out is not None:
            out_arr = self._require_float64_c(out, name="out", ndim=2)
            if out_arr.shape != offspring.shape:
                raise ValueError("out has wrong shape for spea2_generate_offspring.")
            out_arr[:] = offspring
            return out_arr
        return offspring

    def generate_offspring(
        self,
        X: np.ndarray,
        F: np.ndarray,
        params: Mapping[str, object],
        rng: np.random.Generator,
        xl: np.ndarray,
        xu: np.ndarray,
        n_offspring: int | None = None,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        fn = self._cpp_fn("generate_offspring")
        target_offspring = int(n_offspring) if n_offspring is not None else int(X.shape[0])
        if target_offspring <= 0:
            return np.empty((0, X.shape[1]), dtype=np.float64)
        if fn is None:
            ranks, crowd = self.nsga2_ranking(F)
            pressure = _as_int(params.get("selection_pressure"), 2)
            parent_count = max(2, target_offspring)
            parent_idx = self.tournament_selection(ranks, crowd, pressure, rng, n_parents=parent_count)
            offspring = self.sbx_crossover(X[parent_idx], params.get("crossover", {}), rng, xl, xu)
            self.polynomial_mutation(offspring, params.get("mutation", {}), rng, xl, xu)
            return offspring[:target_offspring]

        X_arr = self._require_float64_c(X, name="X", ndim=2)
        F_arr = self._require_float64_c(F, name="F", ndim=2)
        xl_arr = self._require_float64_c(xl, name="xl", ndim=1)
        xu_arr = self._require_float64_c(xu, name="xu", ndim=1)
        operator_cfg = {
            "sbx_prob": _as_float(params.get("sbx_prob"), 0.9),
            "sbx_eta": _as_float(params.get("sbx_eta"), 20.0),
            "pm_prob": _as_float(params.get("pm_prob"), 1.0 / max(1, X_arr.shape[1])),
            "pm_eta": _as_float(params.get("pm_eta"), 20.0),
            "tournament_pressure": _as_int(params.get("tournament_pressure"), 2),
        }
        generated = fn(X_arr, F_arr, target_offspring, xl_arr, xu_arr, operator_cfg, self._seed(rng), out)
        self._mark("generate_offspring")
        return self._require_float64_c(generated, name="generate_offspring output", ndim=2)

    def nsga2_evolve(
        self,
        X: np.ndarray,
        F: np.ndarray,
        evaluate_fn: Any,
        n_gen: int,
        params: Mapping[str, object],
        rng: np.random.Generator,
        xl: np.ndarray,
        xu: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        fn = self._cpp_fn("nsga2_evolve")
        use_native = fn is not None and _env_enabled("VAMOS_ENABLE_CPP_NATIVE_EVOLVE", default=False)
        if not use_native:
            X_curr = X.copy()
            F_curr = F.copy()
            for _ in range(int(n_gen)):
                X_off = self.generate_offspring(X_curr, F_curr, params, rng, xl, xu)
                eval_out = evaluate_fn(X_off)
                F_off = eval_out["F"] if isinstance(eval_out, dict) else getattr(eval_out, "F", eval_out)
                X_curr, F_curr = self.nsga2_survival(X_curr, F_curr, X_off, np.asarray(F_off, dtype=np.float64), X.shape[0])
            return X_curr, F_curr

        X_arr = self._require_float64_c(X, name="X", ndim=2)
        F_arr = self._require_float64_c(F, name="F", ndim=2)
        xl_arr = self._require_float64_c(xl, name="xl", ndim=1)
        xu_arr = self._require_float64_c(xu, name="xu", ndim=1)
        operator_cfg = {
            "sbx_prob": _as_float(params.get("sbx_prob"), 0.9),
            "sbx_eta": _as_float(params.get("sbx_eta"), 20.0),
            "pm_prob": _as_float(params.get("pm_prob"), 1.0 / max(1, X_arr.shape[1])),
            "pm_eta": _as_float(params.get("pm_eta"), 20.0),
            "tournament_pressure": _as_int(params.get("tournament_pressure"), 2),
        }
        self._mark("nsga2_evolve")
        assert fn is not None
        x_new, f_new = fn(X_arr, F_arr, xl_arr, xu_arr, operator_cfg, int(n_gen), self._seed(rng), evaluate_fn)
        return (
            self._require_float64_c(x_new, name="nsga2_evolve X", ndim=2),
            self._require_float64_c(f_new, name="nsga2_evolve F", ndim=2),
        )


__all__ = ["CppBackend"]
