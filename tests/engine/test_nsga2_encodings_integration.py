import numpy as np

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.binary import BinaryKnapsackProblem
from vamos.foundation.problem.integer import IntegerResourceAllocationProblem
from vamos.foundation.problem.mixed import MixedDesignProblem


def _run_nsga2(problem, crossover, mutation, pop_size=10, n_eval=40):
    cfg = (
        NSGAIIConfig()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover(*crossover)
        .mutation(*mutation)
        .selection("tournament", pressure=2)
        .engine("numpy")
    ).fixed()
    algo = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    res = algo.run(problem, termination=("n_eval", n_eval), seed=0)
    assert "F" in res and "X" in res
    assert res["F"].shape[1] == problem.n_obj
    return res


def test_nsga2_binary_runs():
    problem = BinaryKnapsackProblem(n_var=10)
    crossover = ("uniform", {"prob": 0.9})
    mutation = ("bitflip", {"prob": "1/n"})
    res = _run_nsga2(problem, crossover, mutation, pop_size=8, n_eval=32)
    assert res["X"].shape[1] == problem.n_var
    assert np.isfinite(res["F"]).all()


def test_nsga2_integer_runs():
    problem = IntegerResourceAllocationProblem(n_var=8, max_per_task=5)
    crossover = ("uniform", {"prob": 0.9})
    mutation = ("reset", {"prob": "1/n"})
    res = _run_nsga2(problem, crossover, mutation, pop_size=8, n_eval=32)
    assert res["X"].shape[1] == problem.n_var
    assert np.isfinite(res["F"]).all()


def test_nsga2_mixed_runs():
    problem = MixedDesignProblem(n_var=6)
    crossover = ("mixed", {"prob": 0.9})
    mutation = ("mixed", {"prob": "1/n"})
    res = _run_nsga2(problem, crossover, mutation, pop_size=8, n_eval=32)
    assert res["X"].shape[1] == problem.n_var
    assert np.isfinite(res["F"]).all()
