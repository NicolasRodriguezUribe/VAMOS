import numpy as np

from vamos.engine.algorithm.nsgaiii import NSGAIII, associate, nsgaiii_survival
from vamos.engine.algorithm.config import NSGAIIIConfig
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.dtlz import DTLZ2Problem
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _make_config(pop_size=11, prob="1/n"):
    return (
        NSGAIIIConfig()
        .pop_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob=prob, eta=20.0)
        .selection("tournament", pressure=2)
        .reference_directions(divisions=3)
        .engine("numpy")
        .fixed()
    ).to_dict()


def test_nsgaiii_survival_preserves_population_size_with_odd_pop():
    cfg = _make_config(pop_size=11)
    alg = NSGAIII(cfg, kernel=NumPyKernel())
    problem = DTLZ2Problem(n_var=12, n_obj=3)
    result = alg.run(problem, termination=("n_eval", 30), seed=7)
    assert result["X"].shape[0] == 11
    assert result["F"].shape[0] == 11


def test_reference_directions_truncate_when_excess():
    cfg = (
        NSGAIIIConfig()
        .pop_size(6)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .reference_directions(divisions=10)  # generates more than pop_size
        .engine("numpy")
        .fixed()
    ).to_dict()
    alg = NSGAIII(cfg, kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=6)
    result = alg.run(problem, termination=("n_eval", 12), seed=3)
    # Should still respect pop_size=6 even with many reference directions
    assert result["X"].shape[0] == 6


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
    X_sel, F_sel, _, _ = nsgaiii_survival(
        X, F, None, np.empty((0, X.shape[1])), np.empty((0, F.shape[1])), None, cfg["pop_size"], ref_dirs_norm, rng
    )
    # Expect at least one solution per principal direction
    associations, _ = associate(F_sel - F_sel.min(axis=0), ref_dirs_norm)
    assert set(associations) == set(range(n_obj))
