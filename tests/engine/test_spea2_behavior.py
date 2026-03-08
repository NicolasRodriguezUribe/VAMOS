import numpy as np

from vamos.engine.algorithm.config import SPEA2Config
from vamos.engine.algorithm.spea2 import SPEA2
from vamos.engine.algorithm.spea2.helpers import truncate_by_kth_distance
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
