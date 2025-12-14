import numpy as np

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.tuning.evolver import NSGAIITuner
from vamos.engine.tuning.core.parameter_space import AlgorithmConfigSpace, Categorical, Integer, ParameterDefinition


class _DummyResult:
    def __init__(self, F):
        self.F = F
        self.X = np.zeros_like(F)


def _space():
    params = {
        "pop_size": ParameterDefinition(Integer(4, 6)),
        "crossover": ParameterDefinition(Categorical(["sbx"]), fixed_sub_parameters={"prob": 0.9, "eta": 10.0}),
        "mutation": ParameterDefinition(Categorical(["pm"]), fixed_sub_parameters={"prob": 0.1, "eta": 15.0}),
        "selection": ParameterDefinition(Categorical(["tournament"]), fixed_sub_parameters={"pressure": 2}),
    }
    fixed = {"survival": "nsga2", "engine": "numpy"}
    return AlgorithmConfigSpace(NSGAIIConfig, params, fixed_values=fixed)


def test_tuner_respects_budget_limits():
    # Arrange
    space = _space()
    problems = [object()]
    ref_fronts = [None]
    counters = {"runs": 0}

    def _stub_optimize(problem, config, termination, seed, engine):
        counters["runs"] += 1
        F = np.array([[1.0, 1.0]], dtype=float)
        return _DummyResult(F)

    tuner = NSGAIITuner(
        space,
        problems,
        ref_fronts,
        ["hv"],
        max_evals_per_problem=2,
        n_runs_per_problem=1,
        meta_population_size=3,
        meta_max_evals=3,
        max_total_inner_runs=2,
        max_wall_time=5.0,
        seed=1,
        optimize_fn=_stub_optimize,
    )

    # Act
    X, F, configs, diagnostics = tuner.optimize()

    # Assert
    assert diagnostics["n_meta_evals"] <= 3
    assert diagnostics["n_inner_runs"] <= 2
    assert counters["runs"] <= 2
