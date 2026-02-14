from __future__ import annotations

import numpy as np
import pytest

from vamos.engine.tuning import (
    Instance,
    ModelBasedTuner,
    ParamSpace,
    Real,
    TuningTask,
    available_model_based_backends,
)


def _make_task() -> TuningTask:
    return TuningTask(
        name="model_backends_smoke",
        param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}),
        instances=[Instance("p", 2)],
        seeds=[1, 2],
        budget_per_run=4,
        maximize=True,
        aggregator=np.mean,
    )


def _eval_fn(config: dict, ctx) -> float:
    # Peak near x=0.8 and use budget to emulate multi-fidelity signal.
    x = float(config["x"])
    base = 1.0 - abs(x - 0.8)
    return float(base * (ctx.budget / 4.0))


def test_available_model_based_backends_keys():
    backends = available_model_based_backends()
    assert set(backends) == {"optuna", "bohb_optuna", "smac3", "bohb"}


def test_missing_smac3_dependency_raises():
    if available_model_based_backends()["smac3"]:
        pytest.skip("smac3 deps are installed in this environment.")
    tuner = ModelBasedTuner(task=_make_task(), max_trials=1, backend="smac3", seed=0, n_jobs=1)
    with pytest.raises(RuntimeError, match="smac"):
        tuner.run(_eval_fn, verbose=False)


def test_missing_bohb_dependency_raises():
    if available_model_based_backends()["bohb"]:
        pytest.skip("bohb deps are installed in this environment.")
    tuner = ModelBasedTuner(task=_make_task(), max_trials=1, backend="bohb", seed=0, n_jobs=1)
    with pytest.raises(RuntimeError, match="hpbandster"):
        tuner.run(_eval_fn, verbose=False)


def test_optuna_backend_smoke():
    if not available_model_based_backends()["optuna"]:
        pytest.skip("optuna not installed.")
    tuner = ModelBasedTuner(task=_make_task(), max_trials=3, backend="optuna", seed=0, n_jobs=1)
    best, history = tuner.run(_eval_fn, verbose=False)
    assert "x" in best
    assert len(history) >= 1
    assert all("x" in t.config for t in history)


def test_bohb_optuna_backend_smoke():
    if not available_model_based_backends()["bohb_optuna"]:
        pytest.skip("optuna not installed.")
    tuner = ModelBasedTuner(
        task=_make_task(),
        max_trials=3,
        backend="bohb_optuna",
        seed=0,
        n_jobs=1,
        bohb_reduction_factor=2,
    )
    best, history = tuner.run(_eval_fn, verbose=False)
    assert "x" in best
    assert len(history) >= 1
