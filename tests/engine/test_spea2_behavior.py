import numpy as np

from vamos.engine.algorithm.config import SPEA2Config
from vamos.engine.algorithm.spea2 import SPEA2
from vamos.engine.algorithm.spea2.helpers import dominance_matrix, spea2_fitness, truncate_by_kth_distance
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _reference_truncate_by_kth_distance(dist_matrix: np.ndarray, keep: int, k: int) -> np.ndarray:
    active = list(range(dist_matrix.shape[0]))
    neighbor = max(1, int(k))
    while len(active) > keep:
        active_arr = np.asarray(active, dtype=int)
        active_dist = dist_matrix[np.ix_(active_arr, active_arr)]
        active_k = min(neighbor, active_dist.shape[0] - 1)
        density = np.partition(active_dist, kth=active_k, axis=1)[:, active_k]
        order = np.argsort(-density, kind="mergesort")
        active = [active[i] for i in order]
        active.pop()
    return np.asarray(active, dtype=int)


def test_truncate_by_kth_distance_matches_reference_rule():
    rng = np.random.default_rng(5)
    F = rng.random((12, 3))
    dist = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)

    expected = _reference_truncate_by_kth_distance(dist, keep=5, k=2)
    actual = truncate_by_kth_distance(dist, keep=5, k=2)

    assert np.array_equal(expected, actual)


def test_spea2_numba_same_seed_reproducible():
    cfg = (
        SPEA2Config.builder()
        .pop_size(20)
        .archive_size(20)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    ).to_dict()
    problem = ZDT1Problem(n_var=8)

    result_a = SPEA2(cfg, kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 60), seed=11)
    result_b = SPEA2(cfg, kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 60), seed=11)

    assert np.array_equal(result_a["F"], result_b["F"])
    assert np.array_equal(result_a["X"], result_b["X"])


def _spea2_fitness_reference(F: np.ndarray, dom: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    n = F.shape[0]
    if n == 0:
        return np.empty(0), np.empty((0, 0))
    if k is None:
        k = max(1, int(np.sqrt(n)))
    k = min(k, n - 1) if n > 1 else 1

    strength = dom.sum(axis=1)
    raw_fitness = np.zeros(n, dtype=float)
    for i in range(n):
        dominators = np.where(dom[:, i])[0]
        raw_fitness[i] = strength[dominators].sum()

    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            delta = np.linalg.norm(F[i] - F[j])
            dist[i, j] = delta
            dist[j, i] = delta

    if n == 1:
        density = np.array([0.0], dtype=float)
    else:
        density = np.zeros(n, dtype=float)
        for i in range(n):
            sorted_dists = np.sort(dist[i])
            sigma_k = sorted_dists[k] if k < n else sorted_dists[-1]
            density[i] = 1.0 / (sigma_k + 2.0)
    return raw_fitness + density, dist


def test_spea2_fitness_matches_reference_loop_implementation() -> None:
    F = np.array(
        [
            [0.1, 0.9, 0.5],
            [0.4, 0.6, 0.4],
            [0.7, 0.3, 0.6],
            [0.2, 0.8, 0.7],
        ],
        dtype=float,
    )
    dom, _, _ = dominance_matrix(F, None, "none")

    expected_fitness, expected_dist = _spea2_fitness_reference(F, dom, k=2)
    actual_fitness, actual_dist = spea2_fitness(F, dom, k=2)

    np.testing.assert_allclose(actual_fitness, expected_fitness)
    np.testing.assert_allclose(actual_dist, expected_dist)
