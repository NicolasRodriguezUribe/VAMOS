import numpy as np

from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.algorithm.smsemoa import SMSEMOA
from vamos.engine.algorithm.nsgaiii import NSGAIII
from vamos.engine.algorithm.config import MOEADConfig, SMSEMOAConfig, NSGAIIIConfig
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.binary import BinaryKnapsackProblem
from vamos.foundation.problem.integer import IntegerResourceAllocationProblem


def _run_moead(problem, cross, mut, pop_size=12, n_eval=60):
    cfg = (
        MOEADConfig()
        .pop_size(pop_size)
        .neighbor_size(min(5, pop_size))
        .delta(0.9)
        .replace_limit(2)
        .crossover(cross[0], **cross[1])
        .mutation(mut[0], **mut[1])
        .aggregation("tchebycheff")
        .engine("numpy")
    ).fixed()
    algo = MOEAD(cfg.to_dict(), kernel=NumPyKernel())
    res = algo.run(problem, termination=("n_eval", n_eval), seed=0)
    assert "F" in res and res["F"].shape[0] == pop_size
    assert np.isfinite(res["F"]).all()


def _run_smsemoa(problem, cross, mut, pop_size=10, n_eval=40):
    cfg = (
        SMSEMOAConfig()
        .pop_size(pop_size)
        .crossover(cross[0], **cross[1])
        .mutation(mut[0], **mut[1])
        .selection("tournament", pressure=2)
        .reference_point(offset=0.1, adaptive=True)
        .engine("numpy")
    ).fixed()
    algo = SMSEMOA(cfg.to_dict(), kernel=NumPyKernel())
    res = algo.run(problem, termination=("n_eval", n_eval), seed=0)
    assert "F" in res and res["F"].shape[0] == pop_size
    assert np.isfinite(res["F"]).all()


def _run_nsgaiii(problem, cross, mut, pop_size=12, n_eval=60):
    cfg = (
        NSGAIIIConfig()
        .pop_size(pop_size)
        .crossover(cross[0], **cross[1])
        .mutation(mut[0], **mut[1])
        .selection("tournament", pressure=2)
        .reference_directions(path=None)
        .engine("numpy")
    ).fixed()
    algo = NSGAIII(cfg.to_dict(), kernel=NumPyKernel())
    res = algo.run(problem, termination=("n_eval", n_eval), seed=0)
    assert "F" in res and res["F"].shape[0] >= pop_size
    assert np.isfinite(res["F"]).all()


def test_moead_binary_and_integer():
    _run_moead(BinaryKnapsackProblem(n_var=8), ("uniform", {"prob": 0.9}), ("bitflip", {"prob": "1/n"}))
    _run_moead(
        IntegerResourceAllocationProblem(n_var=6, max_per_task=4),
        ("uniform", {"prob": 0.9}),
        ("reset", {"prob": "1/n", "step": 1}),
    )


def test_smsemoa_binary_and_integer():
    _run_smsemoa(BinaryKnapsackProblem(n_var=8), ("uniform", {"prob": 0.9}), ("bitflip", {"prob": "1/n"}))
    _run_smsemoa(
        IntegerResourceAllocationProblem(n_var=6, max_per_task=4),
        ("uniform", {"prob": 0.9}),
        ("reset", {"prob": "1/n", "step": 1}),
    )


def test_nsgaiii_binary_and_integer():
    _run_nsgaiii(BinaryKnapsackProblem(n_var=8), ("uniform", {"prob": 0.9}), ("bitflip", {"prob": "1/n"}))
    _run_nsgaiii(
        IntegerResourceAllocationProblem(n_var=6, max_per_task=4),
        ("uniform", {"prob": 0.9}),
        ("reset", {"prob": "1/n", "step": 1}),
    )
