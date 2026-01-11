"""Test that NSGA-II returns only non-dominated solutions."""

import numpy as np

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def test_nsgaii_result_contains_only_nondominated():
    """Verify that NSGA-II result.F contains only non-dominated solutions.

    The result should not contain any dominated solutions - for each solution,
    there should be no other solution that is better in all objectives.
    """
    pop_size = 20
    cfg = (
        NSGAIIConfig()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .engine("numpy")
        .fixed()
    )
    algorithm = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=10)

    # Run for enough generations to have mixed fronts in population
    result = algorithm.run(problem, termination=("n_eval", pop_size * 3), seed=42)

    F = result["F"]
    n_solutions = F.shape[0]

    # Check that no solution in F is dominated by another solution in F
    for i in range(n_solutions):
        for j in range(n_solutions):
            if i == j:
                continue
            # Check if solution j dominates solution i
            # j dominates i if j <= i in all objectives and j < i in at least one
            dominates = np.all(F[j] <= F[i]) and np.any(F[j] < F[i])
            assert not dominates, f"Solution {i} is dominated by solution {j}. F[{i}]={F[i]}, F[{j}]={F[j]}"


def test_nsgaii_population_key_contains_full_population():
    """Verify that result['population'] contains the full population."""
    pop_size = 15
    cfg = (
        NSGAIIConfig()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .engine("numpy")
        .fixed()
    )
    algorithm = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=8)

    result = algorithm.run(problem, termination=("n_eval", pop_size * 2), seed=123)

    # Full population should be exactly pop_size
    assert "population" in result
    assert result["population"]["F"].shape == (pop_size, problem.n_obj)
    assert result["population"]["X"].shape == (pop_size, problem.n_var)

    # Result F should be <= population F (only non-dominated subset)
    assert result["F"].shape[0] <= pop_size
    assert result["F"].shape[0] > 0
