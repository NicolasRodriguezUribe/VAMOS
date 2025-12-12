import os
from math import comb
from typing import Optional

import numpy as np


def load_or_generate_weight_vectors(
    pop_size: int,
    n_obj: int,
    *,
    path: Optional[str] = None,
    divisions: Optional[int] = None,
) -> np.ndarray:
    """
    Load weight vectors from CSV if available, otherwise generate a simplex-lattice
    design and persist it when a path is provided.
    """
    if path and os.path.exists(path):
        weights = _load_weights(path)
        _assert_valid_weights(weights, n_obj)
        if weights.shape[0] < pop_size:
            raise ValueError(
                f"Weight file '{path}' contains {weights.shape[0]} vectors "
                f"but pop_size={pop_size} requires at least that many."
            )
        return weights[:pop_size]

    weights = _generate_weights(pop_size, n_obj, divisions)

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savetxt(path, weights, delimiter=",")

    return weights


def _load_weights(path: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",")
    arr = np.atleast_2d(arr).astype(float, copy=False)
    return arr


def _assert_valid_weights(weights: np.ndarray, n_obj: int) -> None:
    if weights.ndim != 2:
        raise ValueError("Weight matrix must be 2D.")
    if weights.shape[1] != n_obj:
        raise ValueError(
            f"Expected weight vectors with {n_obj} columns, "
            f"got {weights.shape[1]}."
        )
    if np.any(weights < 0.0):
        raise ValueError("Weight vectors must be non-negative.")
    rows_sum = weights.sum(axis=1)
    # Allow very small numerical drift
    if np.any(np.abs(rows_sum - 1.0) > 1e-6):
        raise ValueError("Each weight vector must sum to 1.")


def _generate_weights(pop_size: int, n_obj: int, divisions: Optional[int]) -> np.ndarray:
    if n_obj < 2:
        # Degenerate single-objective: return uniform weights.
        return np.ones((pop_size, 1), dtype=float)
    if divisions is None:
        divisions = _choose_min_divisions(pop_size, n_obj)
    else:
        max_vectors = _count_lattice_points(n_obj, divisions)
        if max_vectors < pop_size:
            # Automatically increase divisions until we have enough directions.
            fallback = _choose_min_divisions(pop_size, n_obj)
            divisions = max(divisions, fallback)

    weights = _simplex_lattice(n_obj, divisions, limit=pop_size)
    return weights


def _choose_min_divisions(pop_size: int, n_obj: int) -> int:
    divisions = 1
    while _count_lattice_points(n_obj, divisions) < pop_size:
        divisions += 1
    return divisions


def _count_lattice_points(n_obj: int, divisions: int) -> int:
    if divisions < 1:
        raise ValueError("divisions must be >= 1")
    return comb(divisions + n_obj - 1, n_obj - 1)


def _simplex_lattice(n_obj: int, divisions: int, limit: Optional[int]) -> np.ndarray:
    coords = []

    def rec(remaining: int, depth: int, current: list[int]) -> None:
        if limit is not None and len(coords) >= limit:
            return
        if depth == n_obj - 1:
            current.append(remaining)
            coords.append(tuple(current))
            current.pop()
            return
        for value in range(remaining + 1):
            current.append(value)
            rec(remaining - value, depth + 1, current)
            current.pop()
            if limit is not None and len(coords) >= limit:
                return

    rec(divisions, 0, [])
    arr = np.asarray(coords, dtype=float)
    if arr.size == 0:
        raise ValueError("Failed to generate weight vectors.")
    arr /= divisions
    # Numerical guard to keep rows summing to exactly 1
    arr = np.clip(arr, 0.0, 1.0)
    row_sums = arr.sum(axis=1, keepdims=True)
    arr /= row_sums
    return arr
