"""
E2E Test: Full Lifecycle (Optimize -> Save -> Analyze)
"""

import pytest
import pandas as pd

from vamos.foundation.problem.zdt1 import ZDT1Problem
from vamos import OptimizationResult, optimize
from vamos.foundation.metrics import compute_hypervolume
from vamos.engine.algorithm.config import NSGAIIConfig


@pytest.mark.e2e
def test_full_lifecycle_optimize_save_analyze(e2e_workspace):
    """
    Gauntlet Test:
    1. Define Problem (ZDT1)
    2. Optimize (Short run)
    3. Save Results (CSV)
    4. Compute Metrics (Hypervolume)
    """
    # 1. Define Problem
    problem = ZDT1Problem(n_var=10)

    # 2. Optimize (using small budget for speed)
    algo_cfg = (
        NSGAIIConfig.builder()
        .pop_size(20)
        .offspring_size(20)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .build()
    )
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=algo_cfg,
        termination=("n_eval", 100),
        seed=42,
        engine="numpy",
    )

    assert isinstance(result, OptimizationResult)
    assert len(result.F) > 0, "Optimization returned empty front"

    # 3. Save Results
    results_dir = e2e_workspace / "results"
    results_dir.mkdir()
    csv_path = results_dir / "solutions.csv"

    df = pd.DataFrame(result.F, columns=["f1", "f2"])
    df.to_csv(csv_path, index=False)

    assert csv_path.exists(), "CSV file was not created"

    # 4. Analyze (Hypervolume)
    ref_point = [1.1, 1.1]
    hv = compute_hypervolume(result.F, ref_point)
    assert hv >= 0.0, "Hypervolume calculation failed"
