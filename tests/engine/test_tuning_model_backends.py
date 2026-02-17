from __future__ import annotations

from pathlib import Path

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


def test_fidelity_subsampling_scales_instances_and_seeds():
    task = TuningTask(
        name="fidelity_subsample_smoke",
        param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}),
        instances=[
            Instance("zdt1", 2),
            Instance("zdt2", 2),
            Instance("dtlz1", 2),
            Instance("wfg1", 2),
        ],
        seeds=[1, 2, 3, 4],
        budget_per_run=100,
        maximize=True,
        aggregator=np.mean,
    )
    tuner = ModelBasedTuner(
        task=task,
        max_trials=1,
        backend="optuna",
        seed=7,
        n_jobs=1,
        budget_levels=[25, 100],
        fidelity_min_instance_frac=0.5,
        fidelity_min_seed_count=1,
        fidelity_max_seed_count=4,
    )
    ctxs: list[tuple[int, int, int, int | None]] = []

    def eval_fn(_config: dict, ctx) -> float:
        ctxs.append((int(ctx.budget), int(ctx.seed), int(ctx.fidelity_level), ctx.previous_budget))
        return 1.0

    _ = tuner._eval_config_at_budget({"x": 0.5}, eval_fn, budget=25)
    low = list(ctxs)
    assert len(low) == 2  # 2 instances * 1 seed
    assert all(level == 0 for _, _, level, _ in low)
    assert all(prev is None for _, _, _, prev in low)

    ctxs.clear()
    _ = tuner._eval_config_at_budget({"x": 0.5}, eval_fn, budget=100)
    high = list(ctxs)
    assert len(high) == 16  # 4 instances * 4 seeds
    assert all(level == 1 for _, _, level, _ in high)
    assert all(int(prev) == 25 for _, _, _, prev in high)


def test_optuna_storage_resume_and_trace(tmp_path: Path):
    if not available_model_based_backends()["optuna"]:
        pytest.skip("optuna not installed.")

    db_path = tmp_path / "resume_optuna.sqlite3"
    storage_url = f"sqlite:///{db_path.as_posix()}"
    study_name = "resume_optuna_smoke"

    tuner_a = ModelBasedTuner(
        task=_make_task(),
        max_trials=1,
        backend="optuna",
        seed=3,
        n_jobs=1,
        optuna_storage_url=storage_url,
        optuna_study_name=study_name,
        optuna_load_if_exists=True,
    )
    _, history_a = tuner_a.run(_eval_fn, verbose=False)
    assert len(history_a) >= 1
    assert db_path.exists()
    assert any("fidelity_trace" in h.details for h in history_a)

    tuner_b = ModelBasedTuner(
        task=_make_task(),
        max_trials=1,
        backend="optuna",
        seed=3,
        n_jobs=1,
        optuna_storage_url=storage_url,
        optuna_study_name=study_name,
        optuna_load_if_exists=True,
    )
    _, history_b = tuner_b.run(_eval_fn, verbose=False)
    assert len(history_b) >= len(history_a) + 1


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
