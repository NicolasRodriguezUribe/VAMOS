from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vamos.engine.tuning import EvalContext, Instance
from vamos.experiment.cli import tune as tune_cli


def _simple_eval(config: dict[str, float], ctx: EvalContext) -> float:
    _ = ctx
    return float(config["x"])


def test_statistical_finisher_selects_best_candidate():
    candidates = [
        {"score": 0.2, "config": {"x": 0.2}},
        {"score": 0.9, "config": {"x": 0.9}},
    ]
    instances = [Instance(name="zdt1", n_var=10, kwargs={}), Instance(name="zdt2", n_var=10, kwargs={})]
    seeds = [1, 2, 3]

    result = tune_cli._run_statistical_finisher(
        candidates=candidates,
        eval_fn=_simple_eval,
        instances=instances,
        seeds=seeds,
        budget=100,
        aggregator=lambda vals: float(np.mean(vals)),
        alpha=0.05,
        min_blocks=3,
        failure_score=0.0,
        use_friedman=True,
    )

    assert result is not None
    assert float(result["winner_config"]["x"]) == 0.9
    assert int(result["num_candidates"]) == 2
    assert int(result["num_blocks"]) == len(instances) * len(seeds)
    assert any(bool(row["selected"]) for row in result["candidate_rows"])


def test_statistical_finisher_replaces_failed_scores_with_failure_score():
    instances = [Instance(name="zdt1", n_var=10, kwargs={})]
    seeds = [1, 2, 3]

    def flaky_eval(config: dict[str, float], ctx: EvalContext) -> float:
        if int(ctx.seed) == 2:
            raise RuntimeError("boom")
        if int(ctx.seed) == 3:
            return float("nan")
        return float(config["x"])

    result = tune_cli._run_statistical_finisher(
        candidates=[{"score": 0.5, "config": {"x": 0.5}}],
        eval_fn=flaky_eval,
        instances=instances,
        seeds=seeds,
        budget=50,
        aggregator=lambda vals: float(np.mean(vals)),
        alpha=0.05,
        min_blocks=2,
        failure_score=-1.0,
        use_friedman=True,
    )

    assert result is not None
    block_scores = [float(row["score"]) for row in result["block_rows"]]
    assert -1.0 in block_scores
    assert all(np.isfinite(block_scores))
    assert float(result["winner_config"]["x"]) == 0.5


def test_split_instances_suite_stratified_manifest_has_suite():
    instances = [
        Instance(name="zdt1", n_var=10, kwargs={}),
        Instance(name="zdt2", n_var=10, kwargs={}),
        Instance(name="dtlz1", n_var=10, kwargs={}),
        Instance(name="dtlz2", n_var=10, kwargs={}),
        Instance(name="wfg1", n_var=10, kwargs={}),
        Instance(name="wfg2", n_var=10, kwargs={}),
    ]
    train, validation, test, manifest = tune_cli._split_instances(
        instances,
        train_frac=0.6,
        validation_frac=0.2,
        split_seed=42,
        strategy="suite_stratified",
    )

    all_names = {inst.name for inst in instances}
    split_names = {inst.name for inst in train + validation + test}
    assert all_names == split_names
    assert len(validation) >= 1
    assert len(test) >= 1
    assert all("suite" in row for row in manifest)


def test_backend_fallback_switches_to_random(monkeypatch, tmp_path: Path):
    called: dict[str, str] = {}

    def fake_available() -> dict[str, bool]:
        return {"optuna": False, "bohb_optuna": False, "smac3": False, "bohb": False}

    def fake_run_backend(args, task, eval_fn, resolved_jobs):  # noqa: ANN001
        _ = task, eval_fn, resolved_jobs
        called["backend"] = str(args.backend)
        return {"pop_size": 10}, []

    monkeypatch.setattr(tune_cli, "available_model_based_backends", fake_available)
    monkeypatch.setattr(tune_cli, "_run_backend", fake_run_backend)

    tune_cli.main(
        [
            "--instances",
            "zdt1,zdt2,zdt3",
            "--algorithm",
            "nsgaii",
            "--backend",
            "optuna",
            "--backend-fallback",
            "random",
            "--no-run-validation",
            "--no-run-test",
            "--no-run-statistical-finisher",
            "--n-seeds",
            "2",
            "--n-jobs",
            "1",
            "--output-dir",
            str(tmp_path / "tune_out"),
            "--name",
            "fallback_case",
        ]
    )

    assert called["backend"] == "random"


def test_backend_fallback_error_mode_raises(monkeypatch):
    def fake_available() -> dict[str, bool]:
        return {"optuna": False, "bohb_optuna": False, "smac3": False, "bohb": False}

    monkeypatch.setattr(tune_cli, "available_model_based_backends", fake_available)

    with pytest.raises(RuntimeError, match="not available"):
        tune_cli.main(
            [
                "--instances",
                "zdt1,zdt2,zdt3",
                "--algorithm",
                "nsgaii",
                "--backend",
                "optuna",
                "--backend-fallback",
                "error",
                "--no-run-validation",
                "--no-run-test",
                "--no-run-statistical-finisher",
            ]
        )


def test_moea_backend_passes_knowledge_base_flags(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    class _DummyMOEATuner:
        def __init__(self, *, config_space, task, config):  # noqa: ANN001
            _ = config_space, task
            captured["config"] = config

        def run(self, eval_fn, verbose=True):  # noqa: ANN001
            _ = eval_fn, verbose
            return {"pop_size": 20}, []

    monkeypatch.setattr(tune_cli, "MOEATuner", _DummyMOEATuner)

    kb_path = tmp_path / "kb.json"
    args = tune_cli._parse_args(  # noqa: SLF001
        [
            "--algorithm",
            "nsgaii",
            "--backend",
            "moea_tuner",
            "--use-knowledge-base",
            "--knowledge-base-path",
            str(kb_path),
        ]
    )
    algo_space = tune_cli.BUILDERS["nsgaii"]()
    param_space = algo_space.to_param_space()
    task = tune_cli._build_task(args, param_space, budget_per_run=100)  # noqa: SLF001
    tune_cli._run_backend(args, task, _simple_eval, resolved_jobs=1)  # noqa: SLF001

    cfg = captured["config"]
    assert bool(cfg.use_knowledge_base) is True
    assert str(cfg.knowledge_base_path) == str(kb_path)
