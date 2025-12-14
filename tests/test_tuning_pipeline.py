import numpy as np

from vamos.engine.tuning.core.parameter_space import AlgorithmConfigSpace, Categorical, Integer, ParameterDefinition
from vamos.engine.tuning.evolver import TuningPipeline, compute_hyperparameter_importance


class _DummyConfig:
    def __init__(self, label):
        self.label = label

    def to_dict(self):
        return {"label": self.label}


class _StubTuner:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def optimize(self):
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        F = np.array([[0.2, 0.1, 0.05], [0.1, 0.2, 0.3]])
        configs = [_DummyConfig("a"), _DummyConfig("b")]
        diagnostics = {"n_meta_evals": 2, "n_inner_runs": 1, "elapsed_time": 0.01}
        return X, F, configs, diagnostics


class _StubRunner:
    last_tasks = None

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.tasks = None

    def run(self, tasks, export_csv_path=None):
        self.tasks = tasks
        _StubRunner.last_tasks = tasks
        return ["results"]


def _space():
    params = {
        "p1": ParameterDefinition(Integer(1, 3)),
        "p2": ParameterDefinition(Categorical(["a", "b"])),
    }
    return AlgorithmConfigSpace(lambda: _DummyConfig("builder"), params)


def test_pipeline_selects_best_config_and_runs_study():
    # Arrange
    space = _space()
    pipeline = TuningPipeline(
        problems=["toy1", "toy2"],
        base_algorithm="nsgaii",
        config_space=space,
        ref_fronts=[None, None],
        indicators=["hv"],
        tuning_budget={"meta_population_size": 2, "meta_max_evals": 2, "max_evals_per_problem": 2, "n_runs_per_problem": 1},
        seed=0,
        tuner_factory=_StubTuner,
        study_runner_cls=_StubRunner,
    )

    # Act
    pipeline.run_tuning()
    top_cfgs = pipeline.select_top_k(1)
    results = pipeline.run_study(k=1, n_runs=2)

    # Assert
    assert isinstance(results, list)
    assert len(top_cfgs) == 1
    assert top_cfgs[0].label == "b"
    expected_tasks = 1 * 2 * len(pipeline.problems)
    assert _StubRunner.last_tasks is not None
    assert len(_StubRunner.last_tasks) == expected_tasks


def test_importance_computation_orders_features():
    # Arrange
    X = np.array([[0.0, 0.1], [0.5, 0.2], [1.0, 0.3]])
    y = np.array([1.0, 0.5, 0.0])

    # Act
    scores = compute_hyperparameter_importance(X, y, names=["a", "b"])

    # Assert
    assert scores[0][1] >= scores[1][1]
    assert {name for name, _ in scores} == {"a", "b"}
