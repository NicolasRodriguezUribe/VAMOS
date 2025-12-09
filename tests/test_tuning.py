import numpy as np
import pytest

from vamos.algorithm.config import NSGAIIConfig
from vamos.tuning.meta import MetaOptimizationProblem, MetaNSGAII
from vamos.tuning.parameter_space import (
    AlgorithmConfigSpace,
    Boolean,
    Categorical,
    CategoricalInteger,
    Double,
    Integer,
    ParameterDefinition,
)


def _make_nsgaii_space():
    parameters = {
        "pop_size": ParameterDefinition(Integer(10, 20)),
        "crossover": ParameterDefinition(
            Categorical(["sbx", "blx"]),
            sub_parameters={"prob": ParameterDefinition(Double(0.6, 0.9))},
            conditional_sub_parameters={
                "sbx": {"eta": ParameterDefinition(Double(10.0, 30.0))},
                "blx": {"alpha": ParameterDefinition(Double(0.1, 0.8))},
            },
        ),
        "mutation": ParameterDefinition(
            Categorical(["pm"]),
            sub_parameters={
                "prob": ParameterDefinition(Double(0.05, 0.25)),
                "eta": ParameterDefinition(Double(5.0, 25.0)),
            },
        ),
        "selection": ParameterDefinition(
            Categorical(["tournament"]),
            sub_parameters={"pressure": ParameterDefinition(CategoricalInteger([2, 3, 4]))},
        ),
    }
    fixed = {
        "survival": "nsga2",
        "engine": "numpy",
    }
    return AlgorithmConfigSpace(NSGAIIConfig, parameters, fixed_values=fixed)


def test_algorithm_config_space_decodes_conditional_children():
    # Arrange
    space = _make_nsgaii_space()
    vector = np.array([0.5, 0.2, 0.0, 0.75, 0.25, 0.5, 0.25, 0.5, 0.1, 0.9])

    # Act
    cfg = space.decode_vector(vector)

    # Assert
    assert cfg.pop_size == 15
    assert cfg.crossover[0] == "sbx"
    assert cfg.crossover[1]["prob"] == pytest.approx(0.6)
    assert cfg.crossover[1]["eta"] == pytest.approx(25.0)
    assert "alpha" not in cfg.crossover[1]
    assert cfg.mutation[0] == "pm"
    assert cfg.mutation[1]["prob"] == pytest.approx(0.1)
    assert cfg.mutation[1]["eta"] == pytest.approx(15.0)
    assert cfg.selection[0] == "tournament"
    assert cfg.selection[1]["pressure"] == 4
    assert cfg.survival == "nsga2"
    assert cfg.engine == "numpy"


class _ToyProblem:
    def __init__(self, name: str, base: float):
        self.name = name
        self.base = base
        self.n_obj = 2


class _DummyResult:
    def __init__(self, F: np.ndarray):
        self.F = F
        self.X = np.zeros_like(F)


def test_meta_problem_aggregates_runs_with_medians():
    # Arrange
    space = _make_nsgaii_space()
    problems = [_ToyProblem("p1", 1.0), _ToyProblem("p2", 2.0)]
    fixed_x = np.array([0.2] + [0.0] * (space.dim() - 1))
    pop_size = space.decode_vector(fixed_x).pop_size
    calls = []

    def _stub_optimize(problem, config, termination, seed, engine):
        calls.append((problem.name, seed, termination))
        value = problem.base + 0.01 * config.pop_size + (seed % 5) * 0.001
        F = np.array([[value, value + 0.5]], dtype=float)
        return _DummyResult(F)

    meta_rng = np.random.default_rng(7)
    meta_problem = MetaOptimizationProblem(
        space,
        problems,
        [None, None],
        ["placeholder"],
        max_evals_per_problem=3,
        n_runs_per_problem=2,
        engine="numpy",
        meta_rng=meta_rng,
        optimize_fn=_stub_optimize,
    )

    # Act
    objective = meta_problem.evaluate(fixed_x)

    # Assert
    seeds_by_problem = {"p1": [], "p2": []}
    for problem_name, seed, termination in calls:
        assert termination == ("n_eval", 3)
        seeds_by_problem[problem_name].append(seed)
    expected_scores = []
    bases = {p.name: p.base for p in problems}
    for name, seed_list in seeds_by_problem.items():
        run_values = [
            bases[name] + 0.01 * pop_size + (seed % 5) * 0.001 for seed in seed_list
        ]
        expected_scores.append(-np.median(run_values))
    expected_objective = np.mean(expected_scores)
    assert objective.shape == (3,)
    assert objective[0] == pytest.approx(-expected_objective)
    assert objective[1] >= 0.0
    assert objective[2] >= 0.0


def test_meta_nsga2_runs_and_returns_nondominated_set():
    # Arrange
    space = _make_nsgaii_space()
    problem = _ToyProblem("only", 1.5)

    def _stub_optimize(problem, config, termination, seed, engine):
        value = problem.base + 0.01 * config.pop_size + (seed % 3) * 0.001
        F = np.array([[value, value + 0.2]], dtype=float)
        return _DummyResult(F)

    meta_problem = MetaOptimizationProblem(
        space,
        [problem],
        [None],
        ["placeholder"],
        max_evals_per_problem=2,
        n_runs_per_problem=1,
        engine="numpy",
        meta_rng=np.random.default_rng(5),
        optimize_fn=_stub_optimize,
    )
    meta_alg = MetaNSGAII(
        meta_problem,
        population_size=6,
        offspring_size=4,
        max_meta_evals=10,
        seed=11,
    )

    # Act
    X_nd, F_nd, diagnostics = meta_alg.run()

    # Assert
    assert X_nd.shape[1] == space.dim()
    assert F_nd.shape[1] == 3
    assert np.all((X_nd >= 0.0) & (X_nd <= 1.0))
    assert np.isfinite(F_nd).all()
    assert diagnostics["n_meta_evals"] <= 10
