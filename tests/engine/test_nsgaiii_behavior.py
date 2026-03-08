import numpy as np
import pytest

from vamos.engine.algorithm.config import NSGAIIIConfig
from vamos.engine.algorithm.nsgaiii import NSGAIII, associate, nsgaiii_survival
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.dtlz import DTLZ2Problem
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _make_config(pop_size=10, divisions=3, prob="1/n"):
    return (
        NSGAIIIConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob=prob, eta=20.0)
        .selection("tournament", size=2)
        .reference_directions(divisions=divisions)
        .build()
    ).to_dict()


def test_nsgaiii_survival_preserves_population_size_with_odd_pop():
    cfg = _make_config(pop_size=15, divisions=4)
    alg = NSGAIII(cfg, kernel=NumPyKernel())
    problem = DTLZ2Problem(n_var=12, n_obj=3)
    result = alg.run(problem, termination=("max_evaluations", 30), seed=7)
    assert result["X"].shape[0] == 15
    assert result["F"].shape[0] == 15


def test_reference_directions_truncate_when_excess():
    cfg = (
        NSGAIIIConfig.builder()
        .pop_size(6)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .reference_directions(divisions=10)  # generates more than pop_size
        .build()
    ).to_dict()
    alg = NSGAIII(cfg, kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=6)
    with pytest.raises(ValueError, match="pop_size"):
        alg.run(problem, termination=("max_evaluations", 12), seed=3)


def test_association_handles_degenerate_front():
    # Force a degenerate front: all identical objective vectors
    F = np.full((6, 2), 1.0)
    ref_dirs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    ref_dirs_norm = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    associations, distances = associate(F, ref_dirs_norm)
    assert associations.shape[0] == F.shape[0]
    assert np.isfinite(distances).all()


def test_directional_diversity_preserved():
    cfg = _make_config(pop_size=9)
    # Crafted objective values already aligned to distinct ref directions
    F = np.array(
        [
            [0.1, 0.9, 0.9],
            [0.9, 0.1, 0.9],
            [0.9, 0.9, 0.1],
            [0.2, 0.2, 0.8],
            [0.2, 0.8, 0.2],
            [0.8, 0.2, 0.2],
            [0.4, 0.4, 0.6],
            [0.4, 0.6, 0.4],
            [0.6, 0.4, 0.4],
        ]
    )
    X = np.zeros_like(F)
    rng = np.random.default_rng(0)
    n_obj = F.shape[1]
    ref_dirs = np.eye(n_obj)
    ref_dirs_norm = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    # Use public helper function instead of method
    # Simulate with empty offspring arrays for survival test
    ideal = np.full(n_obj, np.inf)
    worst = np.full(n_obj, -np.inf)
    X_sel, F_sel, _, _, _, _, _ = nsgaiii_survival(
        X,
        F,
        None,
        np.empty((0, X.shape[1])),
        np.empty((0, F.shape[1])),
        None,
        cfg["pop_size"],
        ref_dirs_norm,
        rng,
        ideal,
        None,
        worst,
    )
    # Expect at least one solution per principal direction
    associations, _ = associate(F_sel - F_sel.min(axis=0), ref_dirs_norm)
    assert set(associations) == set(range(n_obj))


def test_nsgaiii_survival_matches_backend_rank_path():
    rng = np.random.default_rng(12)
    X = rng.random((18, 5))
    F = rng.random((18, 3))
    X_off = rng.random((18, 5))
    F_off = rng.random((18, 3))
    ref_dirs = rng.random((18, 3))
    ref_dirs_norm = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    ideal = np.full(3, np.inf)
    worst = np.full(3, -np.inf)

    expected = nsgaiii_survival(
        X,
        F,
        None,
        X_off,
        F_off,
        None,
        18,
        ref_dirs_norm,
        np.random.default_rng(7),
        ideal.copy(),
        None,
        worst.copy(),
        kernel=NumPyKernel(),
    )
    actual = nsgaiii_survival(
        X,
        F,
        None,
        X_off,
        F_off,
        None,
        18,
        ref_dirs_norm,
        np.random.default_rng(7),
        ideal.copy(),
        None,
        worst.copy(),
        kernel=NumbaKernel(),
    )

    for left, right in zip(expected[:6], actual[:6], strict=False):
        if left is None or right is None:
            assert left is right
        else:
            assert np.array_equal(left, right)


def test_nsgaiii_numba_same_seed_reproducible():
    cfg = _make_config(pop_size=15, divisions=4)
    problem = DTLZ2Problem(n_var=12, n_obj=3)

    result_a = NSGAIII(cfg, kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 45), seed=3)
    result_b = NSGAIII(cfg, kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 45), seed=3)

    assert np.array_equal(result_a["F"], result_b["F"])
    assert np.array_equal(result_a["X"], result_b["X"])
