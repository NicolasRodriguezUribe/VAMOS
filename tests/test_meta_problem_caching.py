import numpy as np

from vamos.algorithm.config import NSGAIIConfig
from vamos.tuning.evolver import MetaOptimizationProblem
from vamos.tuning.core.parameter_space import AlgorithmConfigSpace, Categorical, Double, Integer, ParameterDefinition


class _DummyResult:
    def __init__(self, F):
        self.F = F
        self.X = np.zeros_like(F)


def _space():
    params = {
        "pop_size": ParameterDefinition(Integer(4, 8)),
        "crossover": ParameterDefinition(Categorical(["sbx"]), fixed_sub_parameters={"prob": 0.9, "eta": 15.0}),
        "mutation": ParameterDefinition(Categorical(["pm"]), fixed_sub_parameters={"prob": 0.1, "eta": 20.0}),
        "selection": ParameterDefinition(Categorical(["tournament"]), fixed_sub_parameters={"pressure": 2}),
    }
    fixed = {"survival": "nsga2", "engine": "numpy"}
    return AlgorithmConfigSpace(NSGAIIConfig, params, fixed_values=fixed)


def test_meta_problem_caches_repeated_configs():
    # Arrange
    space = _space()
    vector = np.full(space.dim(), 0.5)
    calls = {"count": 0}

    def _stub_optimize(problem, config, termination, seed, engine):
        calls["count"] += 1
        F = np.array([[1.0, 1.0]], dtype=float)
        return _DummyResult(F)

    meta_problem = MetaOptimizationProblem(
        space,
        [object()],
        [None],
        ["hv"],
        max_evals_per_problem=2,
        n_runs_per_problem=1,
        engine="numpy",
        meta_rng=np.random.default_rng(0),
        optimize_fn=_stub_optimize,
    )

    # Act
    first = meta_problem.evaluate(vector)
    second = meta_problem.evaluate(vector)

    # Assert
    assert calls["count"] == 1
    assert np.allclose(first, second)
