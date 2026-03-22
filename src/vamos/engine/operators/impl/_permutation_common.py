from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .flags import should_use_numba_variation

PermArray: TypeAlias = NDArray[np.integer[Any]]
PermVec: TypeAlias = PermArray
PermPop: TypeAlias = PermArray
IndexArray: TypeAlias = NDArray[np.integer[Any]]
RNG: TypeAlias = np.random.Generator
CrossoverBuilder: TypeAlias = Callable[[PermVec, PermVec, RNG], tuple[PermVec, PermVec]]
Adjacency: TypeAlias = list[set[int]]

_SWAP_ROWS_JIT: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None] | None = None
_SWAP_ROWS_JIT_DISABLED = False


def _use_numba_variation() -> bool:
    return should_use_numba_variation()


def get_swap_rows_jit() -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None] | None:
    global _SWAP_ROWS_JIT, _SWAP_ROWS_JIT_DISABLED
    if _SWAP_ROWS_JIT_DISABLED:
        return None
    if _SWAP_ROWS_JIT is not None:
        return _SWAP_ROWS_JIT
    if not _use_numba_variation():
        return None
    try:
        from numba import njit
    except ImportError:
        _SWAP_ROWS_JIT_DISABLED = True
        return None

    @njit(cache=True)  # type: ignore[untyped-decorator]
    def _swap_rows_jit(X: np.ndarray, rows: np.ndarray, first: np.ndarray, second: np.ndarray) -> None:
        for idx in range(rows.shape[0]):
            r = rows[idx]
            a = first[idx]
            b = second[idx]
            tmp = X[r, a]
            X[r, a] = X[r, b]
            X[r, b] = tmp

    _SWAP_ROWS_JIT = _swap_rows_jit
    return _SWAP_ROWS_JIT


def random_permutation_population(pop_size: int, n_var: int, rng: RNG) -> PermPop:
    if pop_size <= 0 or n_var <= 0:
        raise ValueError("pop_size and n_var must be positive integers.")
    keys = rng.random((pop_size, n_var))
    return np.argsort(keys, axis=1).astype(np.int32, copy=False)


def ensure_distinct_indices(idx: IndexArray, upper: int, rng: RNG) -> None:
    if idx.size == 0:
        return
    same = idx[:, 0] == idx[:, 1]
    while np.any(same):
        idx[same, 1] = rng.integers(0, upper, size=int(np.count_nonzero(same)))
        same = idx[:, 0] == idx[:, 1]


def ensure_valid_segment(length: int, lo: int, hi: int) -> tuple[int, int]:
    if length < 2:
        return 0, 0
    if hi <= lo:
        hi = lo + 1
    if hi > length:
        hi = length
    return lo, hi


def trim_offspring(offspring: PermPop, n_original: int) -> PermPop:
    return offspring[:n_original] if n_original % 2 else offspring


def two_cut_points(length: int, rng: RNG) -> tuple[int, int]:
    if length < 2:
        return 0, 0
    a = rng.integers(0, length)
    b = rng.integers(0, length - 1)
    if b >= a:
        b += 1
    if a > b:
        a, b = b, a
    return int(a), int(b)


__all__ = [
    "Adjacency",
    "CrossoverBuilder",
    "IndexArray",
    "PermPop",
    "PermVec",
    "RNG",
    "ensure_distinct_indices",
    "ensure_valid_segment",
    "get_swap_rows_jit",
    "random_permutation_population",
    "trim_offspring",
    "two_cut_points",
]
