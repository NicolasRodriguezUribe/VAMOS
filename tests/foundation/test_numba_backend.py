from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from vamos.foundation.kernel.numba_backend import (
    NumbaKernel,
    _compute_crowding_numba,
    _fast_non_dominated_sort_ranks,
    _select_nsga2_indices,
)
from vamos.foundation.kernel.numpy_backend import NumPyKernel


def _reference_sample_tournament_candidates(
    population_size: int,
    pressure: int,
    n_parents: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if pressure == 2:
        candidates = np.empty((n_parents, 2), dtype=np.int64)
        candidates[:, 0] = rng.integers(0, population_size, size=n_parents, dtype=np.int64)
        candidates[:, 1] = rng.integers(0, population_size, size=n_parents, dtype=np.int64)
        duplicates = candidates[:, 0] == candidates[:, 1]
        while np.any(duplicates):
            candidates[duplicates, 1] = rng.integers(0, population_size, size=int(np.sum(duplicates)), dtype=np.int64)
            duplicates = candidates[:, 0] == candidates[:, 1]
        return candidates

    candidates = np.empty((n_parents, pressure), dtype=np.int64)
    for i in range(n_parents):
        candidates[i] = rng.choice(population_size, size=pressure, replace=False)
    return candidates


def _reference_tournament_selection(
    ranks: np.ndarray,
    crowding: np.ndarray,
    pressure: int,
    rng: np.random.Generator,
    n_parents: int,
) -> np.ndarray:
    population_size = ranks.shape[0]
    if pressure <= 0:
        raise ValueError("pressure must be a positive integer")
    if n_parents <= 0 or population_size == 0:
        return np.empty(0, dtype=np.int64)
    if pressure > population_size:
        raise ValueError("pressure cannot exceed population size for tournament selection without replacement")
    if pressure == 1:
        return rng.integers(0, population_size, size=n_parents, dtype=np.int64)

    candidates = _reference_sample_tournament_candidates(population_size, pressure, n_parents, rng)
    tie_breaks = rng.random(candidates.shape)
    winners = np.empty(n_parents, dtype=np.int64)
    for i in range(n_parents):
        best = int(candidates[i, 0])
        best_rank = int(ranks[best])
        best_crowd = float(crowding[best])
        if np.isnan(best_crowd):
            best_crowd = -np.inf
        n_ties = 1
        for j in range(1, pressure):
            cand = int(candidates[i, j])
            cand_rank = int(ranks[cand])
            cand_crowd = float(crowding[cand])
            if np.isnan(cand_crowd):
                cand_crowd = -np.inf
            if cand_rank < best_rank or (cand_rank == best_rank and cand_crowd > best_crowd):
                best = cand
                best_rank = cand_rank
                best_crowd = cand_crowd
                n_ties = 1
            elif cand_rank == best_rank and cand_crowd == best_crowd:
                n_ties += 1
                if tie_breaks[i, j] < (1.0 / n_ties):
                    best = cand
        winners[i] = best
    return winners


@pytest.mark.parametrize(
    "F",
    [
        np.empty((0, 2), dtype=np.float64),
        np.array([[0.2, 0.7]], dtype=np.float64),
        np.array(
            [
                [0.1, 0.9],
                [0.1, 0.9],
                [0.1, 1.0],
                [0.2, 0.8],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.4, 0.9],
                [0.5, 0.5],
            ],
            dtype=np.float64,
        ),
    ],
)
def test_numba_ranking_matches_numpy_on_biobjective_edge_cases(F: np.ndarray) -> None:
    numba_kernel = NumbaKernel()
    numpy_kernel = NumPyKernel()
    ranks_numba, crowd_numba = numba_kernel.nsga2_ranking(F)
    ranks_numpy, crowd_numpy = numpy_kernel.nsga2_ranking(F)
    np.testing.assert_array_equal(ranks_numba, ranks_numpy)
    np.testing.assert_allclose(crowd_numba, crowd_numpy, rtol=0.0, atol=1e-12)


def test_numba_ranking_matches_numpy_on_random_biobjective_cases() -> None:
    numba_kernel = NumbaKernel()
    numpy_kernel = NumPyKernel()
    rng = np.random.default_rng(17)
    for _ in range(25):
        F = rng.integers(0, 9, size=(48, 2)).astype(np.float64)
        ranks_numba, crowd_numba = numba_kernel.nsga2_ranking(F)
        ranks_numpy, crowd_numpy = numpy_kernel.nsga2_ranking(F)
        np.testing.assert_array_equal(ranks_numba, ranks_numpy)
        np.testing.assert_allclose(crowd_numba, crowd_numpy, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("pressure", [1, 2, 4])
def test_numba_tournament_selection_matches_reference(pressure: int) -> None:
    kernel = NumbaKernel()
    ranks = np.array([0, 1, 0, 2, 1, 0, 3, 2, 1, 0], dtype=np.int64)
    crowding = np.array([np.inf, 0.2, np.nan, 0.5, 0.4, np.inf, 0.1, np.nan, 0.7, 0.7], dtype=np.float64)
    n_parents = 32

    rng_expected = np.random.default_rng(1234)
    rng_actual = np.random.default_rng(1234)
    expected = _reference_tournament_selection(ranks, crowding, pressure, rng_expected, n_parents)
    actual = kernel.tournament_selection(ranks, crowding, pressure, rng_actual, n_parents)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("n_obj", [2, 3])
@pytest.mark.parametrize("return_indices", [False, True])
def test_numba_survival_matches_reference_path(n_obj: int, return_indices: bool) -> None:
    numba_kernel = NumbaKernel()
    rng = np.random.default_rng(29)
    X = rng.random((24, 6), dtype=np.float64)
    F = rng.random((24, n_obj), dtype=np.float64)
    X_off = rng.random((24, 6), dtype=np.float64)
    F_off = rng.random((24, n_obj), dtype=np.float64)

    actual = numba_kernel.nsga2_survival(X, F, X_off, F_off, 24, return_indices=return_indices)
    X_comb = np.vstack((X, X_off))
    F_comb = np.vstack((F, F_off))
    ranks = _fast_non_dominated_sort_ranks(F_comb)
    crowding = _compute_crowding_numba(F_comb, ranks)
    selected = _select_nsga2_indices(ranks, crowding, 24)
    if return_indices:
        expected = (X_comb[selected], F_comb[selected], selected)
    else:
        expected = (X_comb[selected], F_comb[selected])

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(a), np.asarray(e), rtol=0.0, atol=1e-12)
