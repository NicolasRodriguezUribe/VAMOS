import pytest

from vamos.engine.algorithm.config import NSGAIIConfig, MOEADConfig
from vamos import OptimizationResult, optimize
from vamos.foundation.exceptions import InvalidAlgorithmError
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _nsgaii_cfg():
    return (
        NSGAIIConfig.builder()
        .pop_size(6)
        .offspring_size(6)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .result_mode("population")
        .build()
    )


def test_optimize_explicit_algorithm_nsga2():
    problem = ZDT1Problem(n_var=6)
    cfg = _nsgaii_cfg()
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        termination=("n_eval", 12),
        seed=1,
        engine="numpy",
    )
    assert isinstance(result, OptimizationResult)
    assert result.F.shape[1] == problem.n_obj
    assert result.X.shape[0] == cfg.pop_size


def test_optimize_explicit_algorithm_moead():
    problem = ZDT1Problem(n_var=6)
    cfg_data = (
        MOEADConfig.builder()
        .pop_size(8)
        .neighbor_size(3)
        .delta(0.9)
        .replace_limit(1)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .build()
    )
    result = optimize(
        problem,
        algorithm="moead",
        algorithm_config=cfg_data,
        termination=("n_eval", 8),
        seed=2,
        engine="numpy",
    )
    assert result.F.shape[0] > 0
    assert result.F.shape[1] == problem.n_obj


def test_optimize_unknown_algorithm_errors():
    problem = ZDT1Problem(n_var=4)
    with pytest.raises(InvalidAlgorithmError, match="Unknown algorithm"):
        optimize(problem, algorithm="unknown_algo", budget=4, pop_size=6)


def test_optimize_rejects_legacy_signature():
    problem = ZDT1Problem(n_var=4)
    with pytest.raises(TypeError, match="algorithm_config"):
        optimize(problem, algorithm="nsgaii", budget=4, algorithm_config={})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        optimize(problem, _nsgaii_cfg(), ("n_eval", 6), 3)  # type: ignore[arg-type]
