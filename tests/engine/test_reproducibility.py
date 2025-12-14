import numpy as np

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.foundation.core.optimize import OptimizeConfig, optimize
from vamos.foundation.problem.zdt1 import ZDT1Problem


def test_optimize_reproducible_with_seed():
    problem = ZDT1Problem(n_var=8)
    cfg = (
        NSGAIIConfig()
        .pop_size(10)
        .offspring_size(10)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .survival("nsga2")
        .engine("numpy")
        .fixed()
    )
    run_cfg = OptimizeConfig(
        problem=problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("n_eval", 30),
        seed=42,
        engine="numpy",
    )
    res1 = optimize(run_cfg)
    res2 = optimize(run_cfg)
    assert np.array_equal(res1.F, res2.F)
