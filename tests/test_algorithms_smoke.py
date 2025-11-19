"""Lightweight smoke tests for the algorithm cores."""

import numpy as np

from vamos.algorithm.config import MOEADConfig, NSGAIIConfig, SMSEMOAConfig
from vamos.algorithm.moead import MOEAD
from vamos.algorithm.nsgaii import NSGAII
from vamos.algorithm.smsemoa import SMSEMOA
from vamos.kernel.numpy_backend import NumPyKernel
from vamos.problem.tsp import TSPProblem
from vamos.problem.zdt1 import ZDT1Problem
from vamos.problem.zdt2 import ZDT2Problem


def test_nsgaii_hv_termination_hits_target():
    pop_size = 10
    cfg = (
        NSGAIIConfig()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .survival("nsga2")
        .engine("numpy")
        .fixed()
    )
    algorithm = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=8)
    hv_term = (
        "hv",
        {
            "target_value": 1e-4,
            "reference_point": [2.0, 2.0],
            "max_evaluations": 40,
        },
    )

    result = algorithm.run(problem, termination=hv_term, seed=3)

    assert result["hv_reached"] is True
    assert result["F"].shape == (pop_size, problem.n_obj)


def test_smsemoa_smoke_runs_with_small_population():
    pop_size = 8
    cfg = (
        SMSEMOAConfig()
        .pop_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .reference_point(offset=0.1, adaptive=True)
        .engine("numpy")
        .fixed()
    )
    algorithm = SMSEMOA(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT2Problem(n_var=6)

    result = algorithm.run(problem, termination=("n_eval", pop_size + 6), seed=4)

    assert result["F"].shape == (pop_size, problem.n_obj)
    assert np.isfinite(result["F"]).all()


def test_moead_smoke_runs_without_weight_files():
    pop_size = 6
    cfg = (
        MOEADConfig()
        .pop_size(pop_size)
        .neighbor_size(3)
        .delta(0.9)
        .replace_limit(2)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .weight_vectors(divisions=6)
        .engine("numpy")
        .fixed()
    )
    algorithm = MOEAD(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=8)

    result = algorithm.run(problem, termination=("n_eval", pop_size + 4), seed=5)

    assert result["F"].shape == (pop_size, problem.n_obj)
    assert np.isfinite(result["F"]).all()


def test_nsgaii_permutation_smoke():
    pop_size = 8
    cfg = (
        NSGAIIConfig()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("ox", prob=0.9)
        .mutation("swap", prob="2/n")
        .selection("tournament", pressure=2)
        .survival("nsga2")
        .engine("numpy")
        .fixed()
    )
    algorithm = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    problem = TSPProblem(n_cities=7)

    result = algorithm.run(problem, termination=("n_eval", pop_size + 6), seed=6)

    X = result["X"]
    assert X.shape == (pop_size, problem.n_var)
    # Every individual should remain a valid permutation.
    expected = np.arange(problem.n_var)
    assert all(np.array_equal(np.sort(row), expected) for row in X)
    assert result["F"].shape == (pop_size, problem.n_obj)
