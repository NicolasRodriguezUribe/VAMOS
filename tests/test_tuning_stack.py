import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from vamos.tuning.param_space import ParamSpace, Real
from vamos.tuning.tuning_task import TuningTask, Instance, EvalContext
from vamos.tuning.random_search_tuner import RandomSearchTuner
from vamos.tuning.racing import RacingTuner
from vamos.tuning.scenario import Scenario
from vamos.tuning.sampler import UniformSampler, ModelBasedSampler
from vamos.tuning.validation import (
    BenchmarkSuite,
    ConfigSpec,
    run_benchmark_suite,
    summarize_benchmark,
)
from vamos.tuning.history import (
    load_history_json,
    select_top_k_trials,
    make_config_specs_from_trials,
    load_top_k_as_config_specs,
    TrialRecord,
)
from vamos.tuning.spec import ExperimentSpec, build_experiment_from_spec


@pytest.fixture
def toy_param_space() -> ParamSpace:
    return ParamSpace(params={"x1": Real(0.0, 1.0), "x2": Real(0.0, 1.0)})


@pytest.fixture
def toy_instances() -> List[Instance]:
    return [
        Instance(name="inst_a", n_var=2),
        Instance(name="inst_b", n_var=2),
    ]


@pytest.fixture
def toy_task(toy_param_space: ParamSpace, toy_instances: List[Instance]) -> TuningTask:
    return TuningTask(
        name="toy",
        param_space=toy_param_space,
        instances=toy_instances,
        seeds=[1, 2],
        budget_per_run=10,
        maximize=True,
        aggregator=np.mean,
    )


@pytest.fixture
def toy_eval_fn() -> Any:
    def _eval(config: Dict[str, Any], ctx: EvalContext) -> float:
        x1 = float(config["x1"])
        x2 = float(config["x2"])
        if ctx.instance.name == "inst_a":
            a1, a2 = 0.3, 0.7
        else:
            a1, a2 = 0.6, 0.4
        base = -((x1 - a1) ** 2 + (x2 - a2) ** 2)
        noise = 1e-6 * ctx.seed
        return base + noise

    return _eval


def test_random_tuner_runs(toy_task: TuningTask, toy_eval_fn: Any) -> None:
    max_trials = 12
    tuner = RandomSearchTuner(task=toy_task, max_trials=max_trials, seed=0)

    best_config, history = tuner.run(toy_eval_fn, verbose=False)

    assert isinstance(best_config, dict) and best_config, "Best config should be a non-empty dict"
    assert set(best_config.keys()) <= set(toy_task.param_space.params.keys())
    assert len(history) == max_trials
    assert all(np.isfinite(t.score) for t in history), "All scores must be finite"


def test_racing_tuner_respects_max_experiments(toy_task: TuningTask, toy_eval_fn: Any) -> None:
    scenario = Scenario(
        max_experiments=30,
        min_survivors=1,
        elimination_fraction=0.5,
        instance_order_random=False,
        seed_order_random=False,
        start_instances=1,
        verbose=False,
        use_statistical_tests=False,
    )
    tuner = RacingTuner(
        task=toy_task,
        scenario=scenario,
        seed=0,
        max_initial_configs=6,
        sampler=UniformSampler(toy_task.param_space),
    )

    best_config, history = tuner.run(toy_eval_fn, verbose=False)

    assert isinstance(best_config, dict) and best_config
    # History scores are aggregated; one per config in this implementation
    assert len(history) <= scenario.max_experiments
    assert all(np.isfinite(t.score) for t in history)


def test_validation_pipeline_runs(
    toy_instances: List[Instance],
    toy_eval_fn: Any,
    toy_param_space: ParamSpace,
) -> None:
    suite = BenchmarkSuite(
        name="toy_suite",
        instances=toy_instances,
        seeds=[1, 2],
        budget_per_run=20,
    )

    configs = [
        ConfigSpec(label="good", config={"x1": 0.3, "x2": 0.7}),
        ConfigSpec(label="mid", config={"x1": 0.5, "x2": 0.5}),
        ConfigSpec(label="bad", config={"x1": 0.0, "x2": 0.0}),
    ]

    report = run_benchmark_suite(toy_eval_fn, suite, configs, maximize=True)
    summaries = summarize_benchmark(report, maximize=True)

    assert len(report.results) == len(configs) * len(toy_instances) * len(suite.seeds)
    assert len(summaries) == len(configs)
    # Verify ranks are strictly increasing
    ranks = [s.rank for s in summaries]
    assert ranks == sorted(ranks), "Summaries should be sorted by rank"
    assert summaries[0].label == "good", "Good config should rank first"
    assert summaries[-1].label == "bad", "Bad config should rank last"


def test_history_utils_roundtrip(tmp_path: Path) -> None:
    trials = [
        {"trial_id": i, "config": {"x1": i * 0.1, "x2": i * 0.1}, "score": float(i)}
        for i in range(5)
    ]
    history_file = tmp_path / "history.json"
    history_file.write_text(json.dumps(trials), encoding="utf-8")

    records = load_history_json(history_file)
    assert len(records) == 5
    top2 = select_top_k_trials(records, k=2, maximize=True, unique_by_config=True)
    assert len(top2) == 2
    assert top2[0].score >= top2[1].score

    specs = make_config_specs_from_trials(top2, label_prefix="test")
    assert len(specs) == 2
    assert all(spec.label.startswith("test") for spec in specs)
    assert specs[0].config == top2[0].config

    specs_via_loader = load_top_k_as_config_specs(history_file, k=2, maximize=True)
    assert len(specs_via_loader) == 2


def test_experiment_spec_and_experiment_run(
    toy_task: TuningTask,
    toy_eval_fn: Any,
) -> None:
    spec_dict = {
        "name": "toy_experiment",
        "tuner_kind": "racing",
        "seed": 123,
        "maximize": True,
        "racing": {
            "max_initial_configs": 5,
            "scenario": {
                "max_experiments": 20,
                "min_survivors": 1,
                "elimination_fraction": 0.5,
                "instance_order_random": False,
                "seed_order_random": False,
                "start_instances": 1,
                "verbose": False,
                "use_statistical_tests": False,
            },
        },
        "sampler": {"type": "uniform"},
        # Keep validation off for speed; could enable with tiny settings if desired
    }

    spec = ExperimentSpec.from_dict(spec_dict)
    experiment = build_experiment_from_spec(spec, toy_task)

    result = experiment.run(toy_eval_fn, verbose=False)

    assert isinstance(result.best_config, dict) and result.best_config
    assert isinstance(result.tuning_history, list) and result.tuning_history
    assert result.benchmark_report is None


def test_model_based_sampler_bias(toy_param_space: ParamSpace) -> None:
    sampler = ModelBasedSampler(
        param_space=toy_param_space,
        exploration_prob=0.1,
        min_samples_to_model=3,
    )
    good_configs = [
        {"x1": 0.9, "x2": 0.9},
        {"x1": 0.85, "x2": 0.95},
        {"x1": 0.92, "x2": 0.88},
    ]
    sampler.update(good_configs)
    rng = np.random.default_rng(0)
    samples = [sampler.sample(rng) for _ in range(100)]
    near = sum(1 for s in samples if s["x1"] > 0.7)
    assert near > 60, "Model-based sampler should bias toward observed good region"
