import numpy as np

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.tuning.evolver import MetaOptimizationProblem
from vamos.engine.tuning.core.parameter_space import AlgorithmConfigSpace, Categorical, Double, Integer, ParameterDefinition


class _DummyResult:
    def __init__(self, F):
        self.F = F
        self.X = np.zeros_like(F)


class _TinyProblem:
    def __init__(self):
        self.name = "tiny"
        self.n_obj = 2


def _space():
    params = {
        "pop_size": ParameterDefinition(Integer(10, 20)),
        "crossover": ParameterDefinition(
            Categorical(["sbx"]),
            fixed_sub_parameters={"prob": 0.9, "eta": 15.0},
        ),
        "mutation": ParameterDefinition(
            Categorical(["pm"]),
            fixed_sub_parameters={"prob": 0.1, "eta": 20.0},
        ),
        "selection": ParameterDefinition(
            Categorical(["tournament"]),
            fixed_sub_parameters={"pressure": 2},
        ),
    }
    fixed = {"survival": "nsga2", "engine": "numpy"}
    return AlgorithmConfigSpace(NSGAIIConfig, params, fixed_values=fixed)


def test_meta_problem_returns_three_objectives():
    # Arrange
    space = _space()
    problem = _TinyProblem()
    vector = np.full(space.dim(), 0.5)

    def _stub_optimize(cfg):
        F = np.array([[1.0, 1.0]], dtype=float)
        return _DummyResult(F)

    meta_problem = MetaOptimizationProblem(
        space,
        [problem],
        [np.array([[1.0, 1.0]])],
        ["hv"],
        max_evals_per_problem=2,
        n_runs_per_problem=2,
        engine="numpy",
        meta_rng=np.random.default_rng(0),
        optimize_fn=_stub_optimize,
    )

    # Act
    objectives = meta_problem.evaluate(vector)

    # Assert
    assert objectives.shape == (3,)
    assert objectives[1] >= 0.0
    assert objectives[2] >= 0.0
