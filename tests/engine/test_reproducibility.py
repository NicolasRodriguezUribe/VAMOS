import numpy as np

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos import optimize
from vamos.foundation.problem.zdt1 import ZDT1Problem


def test_optimize_reproducible_with_seed():
    problem = ZDT1Problem(n_var=8)
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(10)
        .offspring_size(10)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .build()
    )
    res1 = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("n_eval", 30),
        seed=42,
        engine="numpy",
    )
    res2 = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("n_eval", 30),
        seed=42,
        engine="numpy",
    )
    assert np.array_equal(res1.F, res2.F)
