import numpy as np

from vamos.engine.algorithm.config import MOEADConfig, NSGAIIConfig
from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _nsgaii_cfg(pop_size: int) -> NSGAIIConfig:
    return (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .build()
    )


def _moead_cfg(pop_size: int) -> MOEADConfig:
    return (
        MOEADConfig.builder()
        .pop_size(pop_size)
        .batch_size(1)
        .neighbor_size(max(2, min(4, pop_size)))
        .delta(0.9)
        .replace_limit(2)
        .crossover("de", cr=1.0, f=0.5)
        .mutation("pm", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .weight_vectors(divisions=6)
        .build()
    )


def test_nsgaii_warm_start_matches_full_run() -> None:
    pop_size = 6
    budget_stage1 = pop_size * 2
    budget_full = pop_size * 3

    cfg = _nsgaii_cfg(pop_size)
    problem = ZDT1Problem(n_var=4)

    full = NSGAII(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("n_eval", budget_full),
        seed=42,
    )

    stage1 = NSGAII(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("n_eval", budget_stage1),
        seed=42,
    )
    checkpoint = stage1["checkpoint"]

    resumed = NSGAII(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("n_eval", budget_full),
        seed=123,
        checkpoint=checkpoint,
    )

    assert resumed["evaluations"] == budget_full
    assert np.allclose(full["population"]["F"], resumed["population"]["F"])
    assert np.allclose(full["population"]["X"], resumed["population"]["X"])


def test_moead_warm_start_matches_full_run() -> None:
    pop_size = 6
    budget_stage1 = pop_size + 2
    budget_full = pop_size + 4

    cfg = _moead_cfg(pop_size)
    problem = ZDT1Problem(n_var=4)

    full = MOEAD(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("n_eval", budget_full),
        seed=7,
    )

    stage1 = MOEAD(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("n_eval", budget_stage1),
        seed=7,
    )
    checkpoint = stage1["checkpoint"]

    resumed = MOEAD(cfg.to_dict(), kernel=NumPyKernel()).run(
        problem,
        termination=("n_eval", budget_full),
        seed=99,
        checkpoint=checkpoint,
    )

    assert resumed["evaluations"] == budget_full
    assert np.allclose(full["population"]["F"], resumed["population"]["F"])
    assert np.allclose(full["population"]["X"], resumed["population"]["X"])
