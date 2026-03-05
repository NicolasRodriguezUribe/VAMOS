from __future__ import annotations

import numpy as np

from vamos.engine.tuning import Instance, TuningTask
from vamos.engine.tuning.moea_tuner import (
    PHASE_C_ROLES,
    MOEATuner,
    MOEATunerConfig,
    _allocate_phase_experiments,
)
from vamos.engine.tuning.racing.bridge_spaces_nsgaii import build_nsgaii_config_space


def _make_task() -> tuple[MOEATuner, TuningTask]:
    space = build_nsgaii_config_space()
    task = TuningTask(
        name="moea_tuner_test",
        param_space=space.to_param_space(),
        instances=[Instance(name="zdt1", n_var=10, kwargs={"n_obj": 2})],
        seeds=[1],
        budget_per_run=100,
        maximize=True,
        aggregator=np.mean,
    )
    tuner = MOEATuner(
        config_space=space,
        task=task,
        config=MOEATunerConfig(
            max_experiments=9,
            max_initial_configs=5,
            seed=7,
            verbose=False,
        ),
    )
    return tuner, task


def test_allocate_phase_experiments_matches_total():
    assert _allocate_phase_experiments(total=2, split=(0.3, 0.35, 0.35)) == [1, 1, 0]
    assert _allocate_phase_experiments(total=200, split=(0.3, 0.35, 0.35)) == [60, 70, 70]


def test_phase_c_uses_frozen_parents_for_conditional_rates():
    tuner, _ = _make_task()
    full_space = tuner.config_space.to_param_space()
    role_groups = tuner.config_space.group_by_role()

    phase_result = tuner._run_phase(
        phase_name="C (Fine-tuning)",
        active_roles=PHASE_C_ROLES,
        frozen_params={
            "initializer": "random",
            "selection": "tournament",
            "crossover": "sbx",
            "mutation": "gaussian",
        },
        full_param_space=full_space,
        role_groups=role_groups,
        eval_fn=lambda cfg, ctx: float(cfg.get("crossover_eta", 0.0) + cfg.get("gaussian_sigma", 0.0)),
        max_experiments=6,
        budget_per_run=50,
        verbose=False,
    )

    finite_trials = [t for t in phase_result.history if np.isfinite(float(t.score))]
    assert finite_trials
    assert all("crossover_eta" in t.config for t in finite_trials)
    assert all("gaussian_sigma" in t.config for t in finite_trials)


def test_phase_populates_diagnostics_and_small_budget_run():
    tuner, _ = _make_task()
    best, _history = tuner.run(
        lambda cfg, ctx: float(cfg.get("crossover_prob", 0.0)),
        verbose=False,
    )
    assert best
    assert tuner.phase_results
    assert tuner.phase_results[0].diagnostics

    tiny_tuner = MOEATuner(
        config_space=tuner.config_space,
        task=tuner.task,
        config=MOEATunerConfig(
            max_experiments=2,
            max_initial_configs=3,
            seed=5,
            verbose=False,
        ),
    )
    tiny_tuner.run(lambda cfg, ctx: float(cfg.get("crossover_prob", 0.0)), verbose=False)
    assert tiny_tuner.phase_results[-1].history == []
