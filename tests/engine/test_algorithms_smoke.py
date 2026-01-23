"""Lightweight smoke tests for the algorithm cores."""

import numpy as np

from vamos.engine.algorithm.config import (
    MOEADConfig,
    NSGAIIConfig,
    SMSEMOAConfig,
    SPEA2Config,
    IBEAConfig,
    SMPSOConfig,
)
from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.engine.algorithm.smsemoa import SMSEMOA
from vamos.engine.algorithm.spea2 import SPEA2
from vamos.engine.algorithm.ibea import IBEA
from vamos.engine.algorithm.smpso import SMPSO
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.tsp import TSPProblem
from vamos.foundation.problem.zdt1 import ZDT1Problem
from vamos.foundation.problem.zdt2 import ZDT2Problem


def test_nsgaii_hv_termination_hits_target():
    pop_size = 10
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .build()
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
    # Result contains only non-dominated solutions (may be <= pop_size)
    assert result["F"].shape[0] <= pop_size
    assert result["F"].shape[1] == problem.n_obj
    # Full population still available
    assert result["population"]["F"].shape == (pop_size, problem.n_obj)


def test_smsemoa_smoke_runs_with_small_population():
    pop_size = 8
    cfg = (
        SMSEMOAConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("random")
        .reference_point(offset=1.0, adaptive=True)
        .build()
    )
    algorithm = SMSEMOA(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT2Problem(n_var=6)

    result = algorithm.run(problem, termination=("n_eval", pop_size + 6), seed=4)

    # Result contains only non-dominated solutions (may be <= pop_size)
    assert result["F"].shape[0] <= pop_size
    assert result["F"].shape[1] == problem.n_obj
    assert np.isfinite(result["F"]).all()
    # Full population still available
    assert result["population"]["F"].shape == (pop_size, problem.n_obj)


def test_smsemoa_uses_eval_strategy():
    pop_size = 8
    cfg = (
        SMSEMOAConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("random")
        .reference_point(offset=1.0, adaptive=True)
        .build()
    )
    algorithm = SMSEMOA(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT2Problem(n_var=6)

    class CountingBackend:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def evaluate(self, X, problem):  # noqa: ANN001 - protocol-style test double
            from vamos.foundation.eval import EvaluationResult

            self.calls.append(int(X.shape[0]))
            F = np.full((X.shape[0], problem.n_obj), 0.1, dtype=float)
            return EvaluationResult(F=F, G=None)

    backend = CountingBackend()
    algorithm.run(problem, termination=("n_eval", pop_size + 3), seed=4, eval_strategy=backend)

    assert backend.calls[0] == pop_size  # initial population
    assert backend.calls[1:] == [1, 1, 1]  # steady-state offspring evaluations


def test_moead_smoke_runs_without_weight_files():
    pop_size = 6
    cfg = (
        MOEADConfig.builder()
        .pop_size(pop_size)
        .neighbor_size(3)
        .delta(0.9)
        .replace_limit(2)
        .crossover("de", cr=1.0, f=0.5)
        .mutation("pm", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .weight_vectors(divisions=6)
        .build()
    )
    algorithm = MOEAD(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=8)

    result = algorithm.run(problem, termination=("n_eval", pop_size + 4), seed=5)

    # Result contains only non-dominated solutions (may be <= pop_size)
    assert result["F"].shape[0] <= pop_size
    assert result["F"].shape[1] == problem.n_obj
    assert np.isfinite(result["F"]).all()
    # Full population still available
    assert result["population"]["F"].shape == (pop_size, problem.n_obj)


def test_spea2_smoke_runs_with_archive():
    pop_size = 12
    cfg = (
        SPEA2Config.builder()
        .pop_size(pop_size)
        .archive_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .build()
    )
    algorithm = SPEA2(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=6)

    result = algorithm.run(problem, termination=("n_eval", pop_size + 12), seed=7)

    assert result["F"].shape[0] == cfg.archive_size
    assert result["F"].shape[1] == problem.n_obj
    assert np.isfinite(result["F"]).all()


def test_ibea_smoke_indicator_eps():
    pop_size = 10
    cfg = (
        IBEAConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .indicator("eps")
        .kappa(0.05)
        .build()
    )
    algorithm = IBEA(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT2Problem(n_var=6)

    result = algorithm.run(problem, termination=("n_eval", pop_size + 10), seed=8)

    # Result contains only non-dominated solutions (may be <= pop_size)
    assert result["F"].shape[0] <= pop_size
    assert result["F"].shape[1] == problem.n_obj
    assert np.isfinite(result["F"]).all()
    # Full population still available
    assert result["population"]["F"].shape == (pop_size, problem.n_obj)


def test_smpso_smoke_runs():
    pop_size = 14
    cfg = SMPSOConfig.builder().pop_size(pop_size).archive_size(pop_size).mutation("pm", prob="1/n", eta=20.0).build()
    algorithm = SMPSO(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=5)

    result = algorithm.run(problem, termination=("n_eval", pop_size * 2), seed=9)

    assert result["F"].shape[0] > 0
    assert result["F"].shape[1] == problem.n_obj
    assert np.isfinite(result["F"]).all()


def test_nsgaii_with_multiprocessing_eval_strategy():
    pop_size = 10
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .build()
    )
    algorithm = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=6)
    from vamos.foundation.eval.backends import MultiprocessingEvalBackend

    eval_strategy = MultiprocessingEvalBackend(n_workers=2)
    result = algorithm.run(problem, termination=("n_eval", pop_size + 6), seed=10, eval_strategy=eval_strategy)

    # Result contains only non-dominated solutions (may be <= pop_size)
    assert result["F"].shape[0] <= pop_size
    assert result["F"].shape[1] == problem.n_obj
    assert np.isfinite(result["F"]).all()
    # Full population still available
    assert result["population"]["F"].shape == (pop_size, problem.n_obj)


def test_nsgaii_permutation_smoke():
    pop_size = 8
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("ox", prob=0.9)
        .mutation("swap", prob="2/n")
        .selection("tournament", pressure=2)
        .build()
    )
    algorithm = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    problem = TSPProblem(n_cities=7)

    result = algorithm.run(problem, termination=("n_eval", pop_size + 6), seed=6)

    X = result["X"]
    # Result contains only non-dominated solutions (may be <= pop_size)
    assert X.shape[0] <= pop_size
    assert X.shape[1] == problem.n_var
    # Every individual should remain a valid permutation.
    expected = np.arange(problem.n_var)
    assert all(np.array_equal(np.sort(row), expected) for row in X)
    assert result["F"].shape[0] <= pop_size
    assert result["F"].shape[1] == problem.n_obj
    # Full population still available
    assert result["population"]["F"].shape == (pop_size, problem.n_obj)
