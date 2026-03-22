import pytest

from vamos import OptimizationResult, make_problem, optimize
from vamos.engine.algorithm.config import MOEADConfig, NSGAIIConfig
from vamos.foundation.exceptions import ConfigurationError, InvalidAlgorithmError
from vamos.foundation.problem.binary import BinaryKnapsackProblem
from vamos.foundation.problem.tsp import TSPProblem
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _nsgaii_cfg():
    return (
        NSGAIIConfig.builder()
        .pop_size(6)
        .offspring_size(6)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
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
        max_evaluations=12,
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
        .mutation("polynomial", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .build()
    )
    result = optimize(
        problem,
        algorithm="moead",
        algorithm_config=cfg_data,
        max_evaluations=8,
        seed=2,
        engine="numpy",
    )
    assert result.F.shape[0] > 0
    assert result.F.shape[1] == problem.n_obj


def test_optimize_unknown_algorithm_errors():
    problem = ZDT1Problem(n_var=4)
    with pytest.raises(InvalidAlgorithmError, match="Unknown algorithm"):
        optimize(problem, algorithm="unknown_algo", max_evaluations=4, pop_size=6)


def test_optimize_rejects_legacy_signature():
    problem = ZDT1Problem(n_var=4)
    with pytest.raises(ConfigurationError, match="algorithm_config"):
        optimize(problem, algorithm="nsgaii", max_evaluations=4, algorithm_config={})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="unexpected keyword argument 'termination'"):
        optimize(problem, algorithm="nsgaii", termination=("max_evaluations", 6))  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        optimize(problem, _nsgaii_cfg(), ("max_evaluations", 6), 3)  # type: ignore[arg-type]


def test_optimize_resolves_pop_size_consistently() -> None:
    problem = ZDT1Problem(n_var=6)
    pop_size = 10
    max_evaluations = 12

    result_direct = optimize(
        problem,
        algorithm="nsgaii",
        max_evaluations=max_evaluations,
        pop_size=pop_size,
        seed=1,
        engine="numpy",
    )
    cfg = NSGAIIConfig.default(pop_size=pop_size, n_var=problem.n_var)
    result_cfg = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=cfg,
        max_evaluations=max_evaluations,
        seed=1,
        engine="numpy",
    )

    direct_defaults = result_direct.explain_defaults()
    cfg_defaults = result_cfg.explain_defaults()
    assert "resolved_config" in direct_defaults
    assert "resolved_config" in cfg_defaults
    assert direct_defaults["resolved_config"]["pop_size"] == cfg_defaults["resolved_config"]["pop_size"] == pop_size
    assert direct_defaults["resolved_config"]["max_evaluations"] == cfg_defaults["resolved_config"]["max_evaluations"] == max_evaluations


def test_optimize_accepts_max_evaluations() -> None:
    problem = ZDT1Problem(n_var=6)
    result = optimize(
        problem,
        algorithm="nsgaii",
        max_evaluations=12,
        pop_size=6,
        seed=1,
        engine="numpy",
    )
    defaults = result.explain_defaults()
    assert defaults["resolved_config"]["max_evaluations"] == 12


def test_optimize_records_backend_resolution_metadata() -> None:
    problem = ZDT1Problem(n_var=6)
    result = optimize(
        problem,
        algorithm="nsgaii",
        max_evaluations=12,
        pop_size=6,
        seed=1,
    )

    defaults = result.explain_defaults()
    assert defaults["resolved_config"]["engine"] == "numpy"
    assert defaults["resolved_config"]["engine_source"] == "default"
    assert defaults["resolved_config"]["kernel_backend"] == "numpy"
    assert result.meta["engine_source"] == "default"
    assert result.meta["kernel_backend"] == "numpy"


def test_optimize_rejects_termination_keyword() -> None:
    problem = ZDT1Problem(n_var=6)
    with pytest.raises(TypeError, match="unexpected keyword argument 'termination'"):
        optimize(problem, algorithm="nsgaii", pop_size=6, termination=("max_evaluations", 12), seed=1, engine="numpy")  # type: ignore[call-arg]


def test_optimize_auto_rejects_single_objective_problem() -> None:
    problem = make_problem(
        lambda x: [x[0] ** 2],
        n_var=2,
        n_obj=1,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        encoding="real",
    )

    with pytest.raises(ConfigurationError, match="multi-objective"):
        optimize(problem, algorithm="auto", max_evaluations=12, pop_size=6, seed=1)


def test_optimize_permutation_problem_uses_encoding_defaults() -> None:
    problem = TSPProblem(n_cities=8)
    result = optimize(
        problem,
        algorithm="nsgaii",
        max_evaluations=16,
        pop_size=8,
        seed=1,
        engine="numpy",
    )
    assert isinstance(result, OptimizationResult)
    assert result.F is not None
    assert result.X is not None
    assert result.X.shape[1] == problem.n_var


@pytest.mark.parametrize(
    ("problem", "pop_size"),
    [
        (TSPProblem(n_cities=8), 8),
        (BinaryKnapsackProblem(n_var=16), 16),
    ],
)
def test_optimize_moead_non_real_problem_uses_encoding_defaults(problem, pop_size: int) -> None:
    result = optimize(
        problem,
        algorithm="moead",
        max_evaluations=2 * pop_size,
        pop_size=pop_size,
        seed=1,
        engine="numpy",
    )
    assert isinstance(result, OptimizationResult)
    assert result.F is not None
    assert result.X is not None
    assert result.X.shape[1] == problem.n_var
