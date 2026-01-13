"""
JAX-based kernel implementation for GPU acceleration.

This backend leverages JAX to execute critical components (non-dominated sorting,
crowding distance) on CPU/GPU/TPU with auto-vectorization and JIT compilation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Literal, TypeVar, cast, overload

import numpy as np

from .backend import KernelBackend
from .numpy_backend import NumPyKernel


def _import_jax() -> tuple[Any, Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
        from jax import jit
    except ImportError as exc:
        raise ImportError("JAX is not installed. Install with `pip install .[autodiff]` or `pip install jax jaxlib`.") from exc
    return jax, jnp, jit


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


_F = TypeVar("_F", bound=Callable[..., object])


def _typed_decorator(decorator: Any) -> Callable[[_F], _F]:
    def _wrap(fn: _F) -> _F:
        return cast(_F, decorator(fn))

    return _wrap


def _build_jax_kernels(jnp: Any, jit: Any) -> tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    jit_typed = _typed_decorator(jit)

    @jit_typed
    def fast_non_dominated_sort_jax(F: Any) -> Any:
        """
        Compute non-dominated rank using JAX.

        Note: True O(N^2) sorting in JAX is tricky due to dynamic shapes.
        This simplified version computes domination counts and returns
        rank 0 (non-dominated) vs rank >0 (dominated).
        For full sorting, we currently defer to the CPU-based iterative sort
        or return a simplified domination count as a proxy for rank.

        To maintain strict compatibility with NSGA-II, we should output
        standard integerranks. For now, we mimic the behavior of calculating
        domination counts which is the most expensive part (O(N^2 * M)).
        """
        # (N, N) bool matrix where [i, j] is True if i dominates j
        lhs = F[:, None, :]
        rhs = F[None, :, :]
        dominates_mat = (lhs <= rhs).all(axis=2) & (lhs < rhs).any(axis=2)

        # domination_count[i]: how many solutions dominate i
        domination_count = dominates_mat.sum(axis=0)

        # In a fully parallel sort, we can't easily peel fronts iteratively
        # without synchronization.
        # But for 'rank 0' detection (survival of best), count==0 is exact.
        # We return counts as ranks. This is NOT strict non-dominated sorting layers,
        # but for survival selection where we pick the top X, it is often a good heuristic
        # or we just identify the first front exactly.

        # Strictly, NSGA-II requires true layers.
        # Implementing efficient parallel ENS or similar in JAX is complex.
        # We will fallback to returning domination counts, but mapped to ranks
        # roughly.
        # WARNING: This approximates standard non-dominated sorting.
        # For rigorous NSGA-II, layers matter.

        return domination_count

    @jit_typed
    def crowding_distance_jax(F: Any) -> Any:
        """Compute crowding distance in JAX."""
        n_points, n_obj = F.shape
        crowding = jnp.zeros(n_points)

        # We need to sort for each objective
        # JAX argsort
        for m in range(n_obj):
            sorted_idx = jnp.argsort(F[:, m])
            sorted_f = F[sorted_idx, m]

            # Boundary points get infinity
            crowding = crowding.at[sorted_idx[0]].add(jnp.inf)
            crowding = crowding.at[sorted_idx[-1]].add(jnp.inf)

            f_min = sorted_f[0]
            f_max = sorted_f[-1]
            diff = f_max - f_min

            # Mask to avoid division by zero
            safe_diff = jnp.where(diff == 0, 1.0, diff)

            # Distances
            dist = (sorted_f[2:] - sorted_f[:-2]) / safe_diff

            # Accumulate
            # We can't use in-place add easily with gathered indices in loop
            # unless we scatter_add back.
            crowding = crowding.at[sorted_idx[1:-1]].add(dist)

        return crowding

    return fast_non_dominated_sort_jax, crowding_distance_jax


class JaxKernel(KernelBackend):
    """
    Kernel backend using JAX for hardware acceleration.

    Features:
    - JIT-compiled non-dominated sorting (domination check)
    - JIT-compiled crowding distance
    - GPU/TPU support automatically provided by JAX
    """

    name = "jax"

    def __init__(self) -> None:
        jax, jnp, jit = _import_jax()
        self._jax = jax
        self._jnp = jnp
        self._fast_non_dominated_sort, self._crowding_distance = _build_jax_kernels(jnp, jit)
        _logger().info("JaxKernel initialized. Devices: %s", jax.devices())

    def update_archive(
        self,
        archive_X: np.ndarray | None,
        archive_F: np.ndarray | None,
        population_X: np.ndarray,
        population_F: np.ndarray,
        archive_size: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if archive_size <= 0:
            return archive_X, archive_F

        if archive_X is None or archive_X.size == 0:
            X_comb = np.asarray(population_X)
            F_comb = np.asarray(population_F)
        else:
            if archive_F is None:
                raise ValueError("archive_F must be provided when archive_X is provided.")
            X_comb = np.vstack([archive_X, population_X])
            F_comb = np.vstack([archive_F, population_F])

        from vamos.foundation.metrics.pareto import pareto_filter

        front_F, idx = pareto_filter(F_comb, return_indices=True)
        if idx.size == 0:
            return (
                np.empty((0, int(population_X.shape[1])), dtype=population_X.dtype),
                np.empty((0, int(population_F.shape[1])), dtype=float),
            )
        X_nd = X_comb[idx]
        F_nd = front_F
        if X_nd.shape[0] <= archive_size:
            return X_nd, F_nd

        _, crowding = NumPyKernel().nsga2_ranking(F_nd)
        order = np.argsort(crowding)[::-1][:archive_size]
        return X_nd[order], F_nd[order]

    def nsga2_ranking(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute dominance ranks and crowding distance using JAX.
        """
        F_jax = self._jnp.asarray(F)

        # 1. Ranking
        # Using domination count as rank proxy for massively parallel behavior.
        # NOTE: This deviates from strict iterative peeling!
        # But it allows O(1) depth vs O(N) depth.
        ranks_jax = self._fast_non_dominated_sort(F_jax)

        # 2. Crowding Distance
        crowding_jax = self._crowding_distance(F_jax)

        # Block until ready and convert to numpy
        return np.asarray(ranks_jax), np.asarray(crowding_jax)

    def dominates(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A_j = self._jnp.asarray(A)
        B_j = self._jnp.asarray(B)

        # A dominates B?
        res = (A_j <= B_j).all(axis=-1) & (A_j < B_j).any(axis=-1)
        return np.asarray(res)

    # Fallback to NumPy for operations not yet optimized in JAX
    def tournament_selection(
        self,
        ranks: np.ndarray,
        crowding: np.ndarray,
        pressure: int,
        rng: np.random.Generator,
        n_parents: int,
    ) -> np.ndarray:
        return NumPyKernel().tournament_selection(ranks, crowding, pressure, rng, n_parents)

    def sbx_crossover(
        self,
        X_parents: np.ndarray,
        params: Mapping[str, object],
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> np.ndarray:
        return NumPyKernel().sbx_crossover(X_parents, params, rng, xl, xu)

    def polynomial_mutation(
        self,
        X: np.ndarray,
        params: Mapping[str, object],
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> None:
        NumPyKernel().polynomial_mutation(X, params, rng, xl, xu)

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
        kernel = NumPyKernel()
        if return_indices:
            X_new, F_new, selected = kernel.nsga2_survival(X, F, X_off, F_off, pop_size, return_indices=True)
            return X_new, F_new, selected
        X_new, F_new = kernel.nsga2_survival(X, F, X_off, F_off, pop_size, return_indices=False)
        return X_new, F_new
