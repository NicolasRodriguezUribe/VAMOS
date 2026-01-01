import time
import pytest
from typing import Dict, Any
from src.vamos.engine.tuning.racing.core import RacingTuner
from src.vamos.engine.tuning.racing.scenario import Scenario
from src.vamos.engine.tuning.racing.tuning_task import TuningTask, Instance, EvalContext
from src.vamos.engine.tuning.racing.param_space import ParamSpace, Real


def slow_eval_fn(config: Dict[str, Any], ctx: EvalContext) -> float:
    """A mock eval function that sleeps to simulate work."""
    time.sleep(0.5)
    # Simple objective: target x=0.5
    x = config["x"]
    return abs(x - 0.5)


def test_parallel_speedup():
    """
    Verify that n_jobs > 1 allows parallel execution.
    We'll evaluate 10 configs.
    Sequential: 10 * 0.05 = 0.5s.
    Parallel (2 jobs): ~0.25s.
    """
    param_space = ParamSpace({"x": Real("x", 0.0, 1.0)})
    instances = [Instance("inst1", 1)]
    seeds = [0]

    task = TuningTask(name="test_parallel", param_space=param_space, instances=instances, seeds=seeds, budget_per_run=1, maximize=False)

    # 1. Sequential Run
    scenario_seq = Scenario(max_experiments=20, n_jobs=1, verbose=False, start_instances=1, elimination_fraction=0.1)
    tuner_seq = RacingTuner(task, scenario_seq, max_initial_configs=8)

    start = time.time()
    tuner_seq.run(slow_eval_fn)
    dur_seq = time.time() - start

    # 2. Parallel Run
    scenario_par = Scenario(max_experiments=20, n_jobs=2, verbose=False, start_instances=1, elimination_fraction=0.1)
    tuner_par = RacingTuner(task, scenario_par, max_initial_configs=8)

    start = time.time()
    tuner_par.run(slow_eval_fn)
    dur_par = time.time() - start

    # Expect parallel to be faster
    # Note: Overhead might eat gains for such small sleep, but 0.05 * 8 = 0.4s is huge compared to overhead.
    # Parallel should be around 0.2s + overhead.
    # Let's assess if it's at least faster.

    print(f"Sequential: {dur_seq:.4f}s")
    print(f"Parallel: {dur_par:.4f}s")

    # Relaxed assertion to account for potential test env noise/overhead
    # But usually parallel should be significantly faster here.
    assert dur_par < dur_seq


def test_joblib_pickling_lambda():
    """
    Ensure that we catch pickling errors if user passes unpicklable stuff?
    Actually RacingTuner takes a function. If we pass a lambda it might fail with joblib
    unless using cloudpickle (joblib usually handles some lambdas but not all).
    Let's just ensure standard functions work.
    """
    param_space = ParamSpace({"x": Real("x", 0.0, 1.0)})
    instances = [Instance("inst1", 1)]
    seeds = [0]

    task = TuningTask(name="test_pickle", param_space=param_space, instances=instances, seeds=seeds, budget_per_run=1, maximize=False)

    scenario = Scenario(max_experiments=10, n_jobs=2, verbose=False)
    tuner = RacingTuner(task, scenario, max_initial_configs=2)

    # Use a lambda - likely to fail with standard pickle, but joblib (loky) might handle it.
    # If it fails, users should define top-level functions.
    # "AttributeError: Can't pickle local object..."
    # We won't test for failure, just documenting behavior.

    # Let's test that a local function works (joblib sometimes handles this)
    def local_eval(config, ctx):
        return 0.0

    try:
        tuner.run(local_eval)
    except Exception as e:
        pytest.fail(f"Parallel execution failed with local function: {e}")
