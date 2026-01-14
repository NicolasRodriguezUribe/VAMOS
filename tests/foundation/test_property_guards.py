from __future__ import annotations

import numpy as np

from vamos.foundation.constraints.utils import compute_violation, is_feasible
from vamos.foundation.core.hv_stop import compute_hv_reference
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.metrics.pareto import pareto_filter


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def test_pareto_filter_returns_nondominated_points() -> None:
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = int(rng.integers(1, 60))
        m = int(rng.integers(2, 6))
        F = rng.random((n, m))

        front, idx = pareto_filter(F, return_indices=True)

        assert front.shape[0] == idx.shape[0]
        assert front.shape[1] == m
        assert np.all(idx >= 0) and np.all(idx < n)
        assert len(np.unique(idx)) == len(idx)
        assert np.allclose(front, F[idx])

        for i in idx:
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if _dominates(F[j], F[i]):
                    dominated = True
                    break
            assert not dominated


def test_nsga2_ranking_respects_dominance() -> None:
    rng = np.random.default_rng(1)
    kernel = NumPyKernel()
    for _ in range(20):
        n = int(rng.integers(2, 50))
        m = int(rng.integers(2, 5))
        F = rng.random((n, m))

        ranks, crowding = kernel.nsga2_ranking(F)
        assert ranks.shape[0] == n
        assert crowding.shape[0] == n

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if _dominates(F[i], F[j]):
                    assert ranks[i] < ranks[j]


def test_hv_reference_dominates_fronts() -> None:
    rng = np.random.default_rng(2)
    for _ in range(20):
        n_fronts = int(rng.integers(1, 4))
        n_obj = int(rng.integers(2, 6))
        fronts = [rng.random((int(rng.integers(1, 40)), n_obj)) for _ in range(n_fronts)]

        ref = compute_hv_reference(fronts)
        max_vals = np.vstack(fronts).max(axis=0)

        assert ref.shape == (n_obj,)
        assert np.all(ref > max_vals)


def test_violation_matches_feasibility() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(1, 60))
        m = int(rng.integers(1, 6))
        G = rng.normal(size=(n, m))

        feas = is_feasible(G)
        violation = compute_violation(G)

        assert np.all(violation >= 0.0)
        assert np.all((violation == 0.0) == feas)
