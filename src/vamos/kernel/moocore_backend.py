# kernel/moocore_backend.py
from typing import Iterable

import numpy as np

try:
    import moocore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "MooCoreKernel requires the 'moocore' dependency. Install it or switch to a different backend."
    ) from exc

try:  # Optional JIT acceleration for tournament selection
    from numba import njit
except ImportError:  # pragma: no cover - optional dependency
    njit = None

from vamos.algorithm.hypervolume import hypervolume as hv_fn
from .backend import KernelBackend
from .numpy_backend import NumPyKernel as _NumPyKernel, _compute_crowding, _select_nsga2


def _fronts_from_ranks(ranks: np.ndarray):
    if ranks.size == 0:
        return []
    unique_ranks = np.unique(ranks)
    return [np.flatnonzero(ranks == r).tolist() for r in unique_ranks]


def _tournament_winners_numpy(ranks: np.ndarray, crowding: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    n_rows = candidates.shape[0]
    winners = np.empty(n_rows, dtype=np.int64)
    for i in range(n_rows):
        best_idx = candidates[i, 0]
        best_rank = ranks[best_idx]
        best_crowd = crowding[best_idx]
        for j in range(1, candidates.shape[1]):
            idx = candidates[i, j]
            r = ranks[idx]
            if r < best_rank or (r == best_rank and crowding[idx] > best_crowd):
                best_rank = r
                best_crowd = crowding[idx]
                best_idx = idx
        winners[i] = best_idx
    return winners


if njit is not None:  # pragma: no cover - optional speed-up

    @njit(cache=True)
    def _tournament_winners_numba(ranks, crowding, candidates):
        n_rows = candidates.shape[0]
        n_cols = candidates.shape[1]
        winners = np.empty(n_rows, dtype=np.int64)
        for i in range(n_rows):
            best_idx = candidates[i, 0]
            best_rank = ranks[best_idx]
            best_crowd = crowding[best_idx]
            for j in range(1, n_cols):
                idx = candidates[i, j]
                r = ranks[idx]
                if r < best_rank or (r == best_rank and crowding[idx] > best_crowd):
                    best_rank = r
                    best_crowd = crowding[idx]
                    best_idx = idx
            winners[i] = best_idx
        return winners

else:

    def _tournament_winners_numba(ranks, crowding, candidates):
        return _tournament_winners_numpy(ranks, crowding, candidates)


class MooCoreKernel(KernelBackend):
    """
    Backend that delegates non-dominated sorting to moocore (C implementation).
    The rest of the operators reuse the NumPy implementations.
    """

    def __init__(self):
        self._numpy_ops = _NumPyKernel()

    def capabilities(self) -> Iterable[str]:
        return ("c_backend",)

    def quality_indicators(self) -> Iterable[str]:
        return ("hypervolume",)

    def nsga2_ranking(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ranks = moocore.pareto_rank(np.asarray(F, dtype=np.float64, order="C"))
        fronts = _fronts_from_ranks(ranks)
        crowding = _compute_crowding(F, fronts)
        return ranks, crowding

    def tournament_selection(
        self,
        ranks: np.ndarray,
        crowding: np.ndarray,
        pressure: int,
        rng: np.random.Generator,
        n_parents: int,
    ) -> np.ndarray:
        return self._numpy_ops.tournament_selection(
            ranks, crowding, pressure, rng, n_parents
        )

    def sbx_crossover(
        self,
        X_parents: np.ndarray,
        params: dict,
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> np.ndarray:
        return self._numpy_ops.sbx_crossover(X_parents, params, rng, xl, xu)

    def polynomial_mutation(
        self,
        X: np.ndarray,
        params: dict,
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> None:
        self._numpy_ops.polynomial_mutation(X, params, rng, xl, xu)

    def nsga2_survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_comb = np.vstack((X, X_off))
        F_comb = np.vstack((F, F_off))
        ranks = moocore.pareto_rank(np.asarray(F_comb, dtype=np.float64, order="C"))
        fronts = _fronts_from_ranks(ranks)
        crowding = _compute_crowding(F_comb, fronts)
        sel = _select_nsga2(fronts, crowding, pop_size)
        return X_comb[sel], F_comb[sel]

    def hypervolume(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        return hv_fn(points, reference_point)


class MooCoreKernelV2(MooCoreKernel):
    """
    Variant that leverages additional moocore helpers during survival
    to reduce the NumPy workload.
    """

    def nsga2_survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_comb = np.vstack((X, X_off))
        F_comb = np.vstack((F, F_off))
        F_comb = np.asarray(F_comb, dtype=np.float64, order="C")
        ranks = moocore.pareto_rank(F_comb)

        keep_idx = []
        remaining = pop_size
        current_rank = 0
        while remaining > 0:
            idx = np.flatnonzero(ranks == current_rank)
            if idx.size == 0:
                current_rank += 1
                continue

            front_mask = np.zeros(F_comb.shape[0], dtype=bool)
            front_mask[idx] = True
            nondom_mask = moocore.is_nondominated(F_comb[front_mask])
            front_idx = idx[nondom_mask]

            if front_idx.size <= remaining:
                keep_idx.extend(front_idx.tolist())
                remaining -= front_idx.size
            else:
                ref_point = np.max(F_comb, axis=0) + 1.0
                hv_contrib = moocore.hv_contributions(F_comb[front_idx], ref=ref_point)
                order = np.argsort(hv_contrib)[::-1]
                keep_idx.extend(front_idx[order[:remaining]].tolist())
                remaining = 0

            current_rank += 1

        keep = np.array(keep_idx[:pop_size], dtype=int)
        return X_comb[keep], F_comb[keep]


class MooCoreKernelV3(MooCoreKernel):
    """
    Ultra-optimized variant that keeps reusable buffers for combined populations
    and outsources diversity decisions to MooCore whenever possible.
    """

    def __init__(self):
        super().__init__()
        self._X_buffer: np.ndarray | None = None
        self._F_buffer: np.ndarray | None = None
        self._rank_buffer: np.ndarray | None = None
        self._keep_buffer: np.ndarray | None = None

    def _ensure_buffers(self, total: int, n_var: int, n_obj: int, dtype) -> None:
        dtype = np.dtype(dtype)
        if (
            self._X_buffer is None
            or self._X_buffer.shape[0] < total
            or self._X_buffer.shape[1] != n_var
            or self._X_buffer.dtype != dtype
        ):
            self._X_buffer = np.empty((total, n_var), dtype=dtype, order="C")
        if (
            self._F_buffer is None
            or self._F_buffer.shape[0] < total
            or self._F_buffer.shape[1] != n_obj
        ):
            self._F_buffer = np.empty((total, n_obj), dtype=np.float64, order="C")
        if self._rank_buffer is None or self._rank_buffer.shape[0] < total:
            self._rank_buffer = np.empty(total, dtype=np.int32)

    def _combine(self, X_a, F_a, X_b, F_b):
        n_a = X_a.shape[0]
        n_b = X_b.shape[0]
        total = n_a + n_b
        if total == 0:
            raise ValueError("Cannot combine empty populations.")
        n_var = X_a.shape[1] if n_a > 0 else X_b.shape[1]
        n_obj = F_a.shape[1] if F_a.size else F_b.shape[1]
        dtype = X_a.dtype if n_a > 0 else X_b.dtype
        self._ensure_buffers(total, n_var, n_obj, dtype)
        X_view = self._X_buffer[:total]
        F_view = self._F_buffer[:total]
        if n_a > 0:
            np.copyto(X_view[:n_a], X_a)
            np.copyto(F_view[:n_a], F_a, casting="unsafe")
        if n_b > 0:
            np.copyto(X_view[n_a:], X_b)
            np.copyto(F_view[n_a:], F_b, casting="unsafe")
        return X_view, F_view, total

    def _ensure_keep_buffer(self, size: int) -> np.ndarray:
        if self._keep_buffer is None or self._keep_buffer.shape[0] < size:
            self._keep_buffer = np.empty(size, dtype=np.int64)
        return self._keep_buffer[:size]

    def _pick_diverse(self, F_comb: np.ndarray, front_idx: np.ndarray, remaining: int) -> np.ndarray:
        front = F_comb[front_idx]
        if remaining >= front.shape[0]:
            return front_idx
        ref_point = np.max(front, axis=0) + 1.0
        hv_contrib = moocore.hv_contributions(front, ref=ref_point)
        top_idx = np.argpartition(hv_contrib, -remaining)[-remaining:]
        ordered = top_idx[np.argsort(hv_contrib[top_idx])[::-1]]
        return front_idx[ordered]

    def _survive(self, X_a, F_a, X_b, F_b, target_size: int) -> tuple[np.ndarray, np.ndarray]:
        X_comb, F_comb, total = self._combine(X_a, F_a, X_b, F_b)
        ranks = moocore.pareto_rank(F_comb)
        keep = self._ensure_keep_buffer(target_size)
        taken = 0
        current_rank = 0
        while taken < target_size:
            front_idx = np.flatnonzero(ranks == current_rank)
            if front_idx.size == 0:
                current_rank += 1
                continue
            if taken + front_idx.size <= target_size:
                keep[taken : taken + front_idx.size] = front_idx
                taken += front_idx.size
            else:
                remaining = target_size - taken
                selected = self._pick_diverse(F_comb, front_idx, remaining)
                keep[taken : taken + selected.size] = selected
                taken += selected.size
            current_rank += 1
        sel = np.array(keep[:target_size], dtype=int, copy=False)
        return X_comb[sel].copy(), F_comb[sel].copy()

    def nsga2_survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._survive(X, F, X_off, F_off, pop_size)

    def update_archive(
        self,
        archive_X: np.ndarray | None,
        archive_F: np.ndarray | None,
        population_X: np.ndarray,
        population_F: np.ndarray,
        archive_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if archive_size <= 0:
            return archive_X, archive_F
        if archive_X is None or archive_F is None or archive_X.size == 0:
            empty_X = np.empty((0, population_X.shape[1]), dtype=population_X.dtype)
            empty_F = np.empty((0, population_F.shape[1]), dtype=population_F.dtype)
            archive_X, archive_F = empty_X, empty_F
        return self._survive(archive_X, archive_F, population_X, population_F, archive_size)


class MooCoreKernelV4(MooCoreKernelV3):
    """
    Further optimized variant introducing double buffers, rank-sorted fronts,
    and a crowding fallback for large/high-dimensional fronts to limit HV calls.
    """

    CROWDING_DIM_THRESHOLD = 3
    HV_SIZE_THRESHOLD = 256

    def __init__(self):
        super().__init__()
        self._X_output: np.ndarray | None = None
        self._F_output: np.ndarray | None = None
        self._stats = {"hv_calls": 0, "crowding_fallbacks": 0, "buffer_resizes": 0}

    def _ensure_output_buffers(self, size: int, n_var: int, n_obj: int, dtype) -> None:
        dtype = np.dtype(dtype)
        if (
            self._X_output is None
            or self._X_output.shape[0] < size
            or self._X_output.shape[1] != n_var
            or self._X_output.dtype != dtype
        ):
            self._X_output = np.empty((size, n_var), dtype=dtype, order="C")
            self._stats["buffer_resizes"] += 1
        if (
            self._F_output is None
            or self._F_output.shape[0] < size
            or self._F_output.shape[1] != n_obj
        ):
            self._F_output = np.empty((size, n_obj), dtype=np.float64, order="C")

    @staticmethod
    def _crowding_single(front: np.ndarray) -> np.ndarray:
        if front.shape[0] == 0:
            return np.empty(0, dtype=float)
        fronts = [list(range(front.shape[0]))]
        return _compute_crowding(front, fronts)

    def _select_partial_front(
        self, F_comb: np.ndarray, front_idx: np.ndarray, remaining: int
    ) -> np.ndarray:
        front = F_comb[front_idx]
        use_crowding = (
            front.shape[1] > self.CROWDING_DIM_THRESHOLD
            or front_idx.size > self.HV_SIZE_THRESHOLD
        )
        if use_crowding:
            self._stats["crowding_fallbacks"] += 1
            crowded = self._crowding_single(front)
            order = np.argsort(crowded)[::-1][:remaining]
        else:
            self._stats["hv_calls"] += 1
            ref_point = np.max(front, axis=0) + 1.0
            hv_contrib = moocore.hv_contributions(front, ref=ref_point)
            order = np.argsort(hv_contrib)[::-1][:remaining]
        return front_idx[order]

    def _survive(
        self, X_a, F_a, X_b, F_b, target_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        X_comb, F_comb, total = self._combine(X_a, F_a, X_b, F_b)
        ranks = moocore.pareto_rank(F_comb)
        order = np.argsort(ranks, kind="mergesort")
        keep = self._ensure_keep_buffer(target_size)
        taken = 0
        idx = 0
        while taken < target_size and idx < total:
            rank_value = ranks[order[idx]]
            start = idx
            while idx < total and ranks[order[idx]] == rank_value:
                idx += 1
            front_idx = order[start:idx]
            if taken + front_idx.size <= target_size:
                keep[taken : taken + front_idx.size] = front_idx
                taken += front_idx.size
            else:
                remaining = target_size - taken
                selected = self._select_partial_front(F_comb, front_idx, remaining)
                keep[taken : taken + selected.size] = selected
                taken += selected.size
        sel = keep[:target_size]
        n_var = X_comb.shape[1]
        n_obj = F_comb.shape[1]
        self._ensure_output_buffers(target_size, n_var, n_obj, X_comb.dtype)
        np.copyto(self._X_output[:target_size], X_comb[sel])
        np.copyto(self._F_output[:target_size], F_comb[sel])
        return self._X_output[:target_size], self._F_output[:target_size]


class _IncrementalArchive:
    """
    Lightweight incremental archive that inserts candidates one-by-one,
    removing dominated points and trimming by leveraging the kernel's
    diversity selector when capacity is exceeded.
    """

    def __init__(self, kernel: MooCoreKernelV4, capacity: int, n_var: int, n_obj: int, dtype):
        self.kernel = kernel
        self.capacity = int(capacity)
        self._dtype = np.dtype(dtype)
        self._X = np.empty((0, n_var), dtype=self._dtype)
        self._F = np.empty((0, n_obj), dtype=float)

    def is_empty(self) -> bool:
        return self._X.size == 0

    def bootstrap(self, archive_X: np.ndarray, archive_F: np.ndarray) -> None:
        self._X = np.asarray(archive_X, dtype=self._dtype, order="C").copy()
        self._F = np.asarray(archive_F, dtype=float, order="C").copy()

    def _dominates(self, candidates: np.ndarray, f: np.ndarray) -> np.ndarray:
        if candidates.size == 0:
            return np.zeros(0, dtype=bool)
        less_equal = candidates <= f
        strictly_less = candidates < f
        return np.all(less_equal, axis=1) & np.any(strictly_less, axis=1)

    def _is_dominated(self, f: np.ndarray) -> bool:
        return self._dominates(self._F, f).any()

    def _trim(self) -> None:
        if self._F.shape[0] <= self.capacity:
            return
        idx = np.arange(self._F.shape[0], dtype=int)
        selected = self.kernel._select_partial_front(self._F, idx, self.capacity)
        self._X = self._X[selected]
        self._F = self._F[selected]

    def update(self, pop_X: np.ndarray, pop_F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for i in range(pop_X.shape[0]):
            f = np.asarray(pop_F[i], dtype=float)
            if self._is_dominated(f):
                continue
            dominating = self._dominates(self._F, f)
            if dominating.any():
                keep = ~dominating
                self._X = self._X[keep]
                self._F = self._F[keep]
            self._X = np.vstack((self._X, np.asarray(pop_X[i], dtype=self._dtype)))
            self._F = np.vstack((self._F, f))
            self._trim()
        return self._X, self._F

    def contents(self) -> tuple[np.ndarray, np.ndarray]:
        return self._X, self._F


class MooCoreKernelV5(MooCoreKernelV4):
    """
    Experimental kernel that adds Numba-accelerated tournament selection
    and an incremental archive manager on top of the V4 survival routines.
    """

    def __init__(self):
        super().__init__()
        self._archive_manager: _IncrementalArchive | None = None

    def tournament_selection(
        self,
        ranks: np.ndarray,
        crowding: np.ndarray,
        pressure: int,
        rng: np.random.Generator,
        n_parents: int,
    ) -> np.ndarray:
        return self._numpy_ops.tournament_selection(
            ranks, crowding, pressure, rng, n_parents
        )

    def update_archive(
        self,
        archive_X: np.ndarray | None,
        archive_F: np.ndarray | None,
        population_X: np.ndarray,
        population_F: np.ndarray,
        archive_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if archive_size <= 0:
            return archive_X, archive_F
        if (
            self._archive_manager is None
            or self._archive_manager.capacity != archive_size
            or self._archive_manager._X.shape[1] != population_X.shape[1]
        ):
            self._archive_manager = _IncrementalArchive(
                self, archive_size, population_X.shape[1], population_F.shape[1], population_X.dtype
            )
            if archive_X is not None and archive_X.size:
                self._archive_manager.bootstrap(archive_X, archive_F)
        elif archive_X is not None and archive_X.size and self._archive_manager.is_empty():
            self._archive_manager.bootstrap(archive_X, archive_F)

        updated_X, updated_F = self._archive_manager.update(population_X, population_F)
        return updated_X, updated_F
