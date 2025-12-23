
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.foundation.core.optimize import OptimizeConfig, optimize, OptimizationResult
from vamos.foundation.problem.zdt1 import ZDT1Problem
import numpy as np

def _nsgaii_cfg():
    return (
        NSGAIIConfig()
        .pop_size(6)
        .offspring_size(6)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .survival("nsga2")
        .engine("numpy")
        .result_mode("population")
        .fixed()
    )

def test_optimize_explicit_algorithm_nsga2():
    problem = ZDT1Problem(n_var=6)
    cfg = OptimizeConfig(
        problem=problem,
        algorithm="nsgaii",
        algorithm_config=_nsgaii_cfg(),
        termination=("n_eval", 12),
        seed=1,
        engine="numpy",
    )
    print(f"Config result_mode: {cfg.algorithm_config.result_mode}")
    print("Running optimize...")
    result = optimize(cfg)
    print("Optimize finished.")
    
    assert isinstance(result, OptimizationResult)
    print(f"Result X shape: {result.X.shape}")
    print(f"Result F shape: {result.F.shape}")
    
    if result.X.shape[0] != 6:
        print(f"FAILURE: Expected 6, got {result.X.shape[0]}")
    else:
        print("SUCCESS: Got 6.")

if __name__ == "__main__":
    test_optimize_explicit_algorithm_nsga2()
