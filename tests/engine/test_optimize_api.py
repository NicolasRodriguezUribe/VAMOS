import pytest

from vamos.engine.algorithm.config import NSGAIIConfig, MOEADConfig
from vamos.experiment.optimize import OptimizeConfig, OptimizationResult, optimize
from vamos.foundation.problem.zdt1 import ZDT1Problem


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
    result = optimize(cfg)
    assert isinstance(result, OptimizationResult)
    assert result.F.shape[1] == problem.n_obj
    assert result.X.shape[0] == cfg.algorithm_config.pop_size


def test_optimize_explicit_algorithm_moead():
    problem = ZDT1Problem(n_var=6)
    cfg_data = (
        MOEADConfig()
        .pop_size(8)
        .neighbor_size(3)
        .delta(0.9)
        .replace_limit(1)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .engine("numpy")
        .fixed()
    )
    cfg = OptimizeConfig(
        problem=problem,
        algorithm="moead",
        algorithm_config=cfg_data,
        termination=("n_eval", 8),
        seed=2,
        engine="numpy",
    )
    result = optimize(cfg)
    assert result.F.shape[0] > 0
    assert result.F.shape[1] == problem.n_obj


def test_optimize_unknown_algorithm_errors():
    problem = ZDT1Problem(n_var=4)
    cfg = OptimizeConfig(
        problem=problem,
        algorithm="unknown_algo",
        algorithm_config=_nsgaii_cfg(),
        termination=("n_eval", 4),
        seed=0,
    )
    with pytest.raises(ValueError, match="Unknown algorithm"):
        optimize(cfg)


def test_optimize_rejects_legacy_signature():
    problem = ZDT1Problem(n_var=4)
    cfg = _nsgaii_cfg()
    with pytest.raises(TypeError, match="OptimizeConfig"):
        optimize(problem)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        optimize(problem, cfg, ("n_eval", 6), 3)  # type: ignore[arg-type]
