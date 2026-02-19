"""Tests for new racing features: convergence detection and checkpoint I/O."""

import tempfile
from pathlib import Path

import numpy as np

from vamos.engine.tuning.racing.core import RacingTuner
from vamos.engine.tuning.racing.io import load_checkpoint, save_checkpoint
from vamos.engine.tuning.racing.param_space import ParamSpace, Real
from vamos.engine.tuning.racing.scenario import Scenario
from vamos.engine.tuning.racing.tuning_task import Instance, TuningTask


def test_convergence_detection_stops_early():
    """Test that racing stops when score stagnates for convergence_window stages."""
    task = TuningTask(
        name="demo",
        param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}),
        instances=[Instance("p", 2)],
        seeds=[0, 1, 2, 3, 4, 5],
        budget_per_run=1,
        maximize=False,
        aggregator=np.mean,
    )
    scenario = Scenario(
        max_experiments=1000,
        min_survivors=1,
        elimination_fraction=0.01,
        start_instances=1,
        use_statistical_tests=False,
        instance_order_random=False,
        seed_order_random=False,
        convergence_window=3,
        convergence_threshold=0.05,
    )
    tuner = RacingTuner(task=task, scenario=scenario, seed=0, max_initial_configs=3)

    call_count = [0]

    def eval_fn(config, ctx):
        call_count[0] += 1
        # Return constant score to trigger convergence
        return 0.5

    tuner.run(eval_fn, verbose=False)

    # Should stop early due to convergence
    assert tuner._stage_index < len(tuner._schedule), "Should have stopped early"
    assert len(tuner._best_score_history) >= 3, "Should have tracked at least convergence_window scores"


def test_convergence_disabled_by_default():
    """Test that convergence detection is disabled when window is 0."""
    task = TuningTask(
        name="demo",
        param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}),
        instances=[Instance("p", 2)],
        seeds=[0, 1],
        budget_per_run=1,
        maximize=False,
        aggregator=np.mean,
    )
    scenario = Scenario(
        max_experiments=100,
        min_survivors=1,
        elimination_fraction=0.01,
        start_instances=1,
        use_statistical_tests=False,
        instance_order_random=False,
        seed_order_random=False,
        # Default: convergence_window=0
    )
    tuner = RacingTuner(task=task, scenario=scenario, seed=0, max_initial_configs=3)

    def eval_fn(config, ctx):
        return 0.5

    tuner.run(eval_fn, verbose=False)

    # Should run all stages since convergence is disabled
    assert tuner._stage_index == len(tuner._schedule)


def test_convergence_not_triggered_by_high_variance_plateau():
    """Do not declare convergence when recent best scores are too volatile."""
    task = TuningTask(
        name="demo",
        param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}),
        instances=[Instance("p", 2)],
        seeds=[0],
        budget_per_run=1,
        maximize=True,
        aggregator=np.mean,
    )
    scenario = Scenario(
        max_experiments=100,
        convergence_window=4,
        convergence_threshold=0.05,
        use_statistical_tests=False,
    )
    tuner = RacingTuner(task=task, scenario=scenario, seed=0, max_initial_configs=2)

    # Improvement from oldest to newest is small/negative, but variance is large.
    tuner._best_score_history = [1.0, 0.75, 1.25, 0.98]

    assert tuner._check_convergence() is False


def test_checkpoint_save_load_roundtrip():
    """Test checkpoint save and load preserves data."""
    best_configs = [{"x": 0.5, "pop_size": 100}]
    elite_archive = [{"config": {"x": 0.5}, "score": 0.123}]
    metadata = {"stage_index": 10, "num_experiments": 500}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "checkpoint.json"

        save_checkpoint(best_configs, elite_archive, path, metadata=metadata)

        loaded = load_checkpoint(path)

        assert loaded["best_configs"] == best_configs
        assert loaded["elite_archive"] == elite_archive
        assert loaded["metadata"] == metadata


def test_warm_start_with_initial_configs():
    """Test that initial_configs parameter works as warm start."""
    task = TuningTask(
        name="demo",
        param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}),
        instances=[Instance("p", 2)],
        seeds=[0],
        budget_per_run=1,
        maximize=False,
        aggregator=np.mean,
    )
    scenario = Scenario(
        max_experiments=10,
        min_survivors=1,
        elimination_fraction=0.5,
        use_statistical_tests=False,
    )

    warm_start = [{"x": 0.42}, {"x": 0.84}]

    tuner = RacingTuner(
        task=task,
        scenario=scenario,
        seed=0,
        max_initial_configs=5,
        initial_configs=warm_start,
    )

    def eval_fn(config, ctx):
        return config["x"]  # Lower x is better

    best_cfg, history = tuner.run(eval_fn, verbose=False)

    # Check that warm start configs were included
    trial_x_values = [t.config["x"] for t in history]
    assert 0.42 in trial_x_values, "Warm start config should be in history"
    assert 0.84 in trial_x_values, "Warm start config should be in history"
