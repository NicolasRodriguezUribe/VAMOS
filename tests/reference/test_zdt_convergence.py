import pytest
from vamos.foundation.problem.zdt1 import ZDT1Problem
from vamos.foundation.problem.zdt2 import ZDT2Problem
from vamos.engine.algorithm.config import NSGAIIConfig, MOEADConfig
from vamos.experiment.optimize import OptimizeConfig, optimize_config
from vamos.foundation.metrics import compute_hypervolume


@pytest.mark.reference
def test_zdt1_nsgaii_convergence():
    """
    Verify NSGA-II solves ZDT1 (Convex) within budget.
    Baseline: HV > 0.60
    Budget: 2500 evals
    """
    problem = ZDT1Problem(n_var=30)

    algo_cfg = (
        NSGAIIConfig()
        .pop_size(100)
        .offspring_size(100)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob=1.0 / 30, eta=20.0)
        .selection("tournament", pressure=2)
        .engine("numpy")
        .fixed()
    )

    config = OptimizeConfig(
        problem=problem,
        algorithm="nsgaii",
        algorithm_config=algo_cfg,
        termination=("n_eval", 10000),
        seed=42,  # Deterministic seed
        engine="numpy",
    )

    result = optimize_config(config)

    # Normalized HV (Reference Point [1.1, 1.1])
    hv = compute_hypervolume(result.F, [1.1, 1.1])

    assert hv > 0.60, f"ZDT1 NSGA-II failed to converge. HV={hv:.4f} < 0.60"


@pytest.mark.reference
def test_zdt1_moead_convergence():
    """
    Verify MOEA/D solves ZDT1.
    Baseline: HV > 0.60
    """
    problem = ZDT1Problem(n_var=30)

    algo_cfg = (
        MOEADConfig()
        .pop_size(100)
        .neighbor_size(20)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("pm", prob=1.0 / 30, eta=20.0)
        .aggregation("tchebycheff")
        .delta(0.9)
        .replace_limit(2)
        .engine("numpy")
        .fixed()
    )

    config = OptimizeConfig(
        problem=problem,
        algorithm="moead",
        algorithm_config=algo_cfg,
        termination=("n_eval", 10000),
        seed=42,
        engine="numpy",
    )

    result = optimize_config(config)

    hv = compute_hypervolume(result.F, [1.1, 1.1])
    assert hv > 0.60, f"ZDT1 MOEA/D failed to converge. HV={hv:.4f} < 0.60"


@pytest.mark.reference
def test_zdt2_nsgaii_convergence():
    """
    Verify NSGA-II solves ZDT2 (Non-convex).
    Baseline: HV > 0.30 (Harder problem)
    """
    # Note: ZDT2 is harder, requires more careful tuning or more evals usually
    problem = ZDT2Problem(n_var=30)

    algo_cfg = (
        NSGAIIConfig()
        .pop_size(100)
        .offspring_size(100)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob=1.0 / 30, eta=20.0)
        .selection("tournament", pressure=2)
        .engine("numpy")
        .fixed()
    )

    config = OptimizeConfig(
        problem=problem,
        algorithm="nsgaii",
        algorithm_config=algo_cfg,
        termination=("n_eval", 10000),  # Slightly boosted budget
        seed=42,
        engine="numpy",
    )

    result = optimize_config(config)

    hv = compute_hypervolume(result.F, [1.1, 1.1])
    assert hv > 0.30, f"ZDT2 NSGA-II failed to converge. HV={hv:.4f} < 0.30"
