from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from vamos.assist.apply import apply_plan
from vamos.assist.plan import create_plan
from vamos.assist.run import make_resolved_config, run_plan, run_with_config_path, select_config_path
from vamos.engine.config.spec import validate_experiment_spec
from vamos.experiment.cli.args import build_parser, build_pre_parser
from vamos.experiment.cli.loaders import load_spec_defaults
from vamos.experiment.cli.spec_args import parser_spec_keys
from vamos.foundation.core.experiment_config import ExperimentConfig


def _allowed_overrides() -> set[str]:
    default_config = ExperimentConfig()
    parser = build_parser(
        default_config=default_config,
        pre_parser=build_pre_parser(),
        spec_defaults=load_spec_defaults(None),
    )
    return parser_spec_keys(parser)


def _materialized_plan(tmp_path: Path) -> tuple[Path, Path]:
    plan_dir = create_plan(
        prompt="Run plan smoke test.",
        template="demo",
        problem_type="real",
        out_dir=tmp_path / "plan",
    )
    project_dir = apply_plan(plan_dir, out_dir=tmp_path / "project")
    (plan_dir / "project").mkdir(parents=True, exist_ok=True)
    (plan_dir / "project" / "config.json").write_text((project_dir / "config.json").read_text(encoding="utf-8"), encoding="utf-8")
    return plan_dir, project_dir


def test_select_config_path_prefers_project_config(tmp_path: Path) -> None:
    plan_dir, _ = _materialized_plan(tmp_path)
    selected = select_config_path(plan_dir)
    assert selected == plan_dir / "project" / "config.json"


def test_make_resolved_config_smoke_is_valid_and_sets_output_root(tmp_path: Path) -> None:
    plan_dir, project_dir = _materialized_plan(tmp_path)
    base_config = json.loads((project_dir / "config.json").read_text(encoding="utf-8"))

    run_dir = plan_dir / "runs" / "run_smoke"
    resolved = make_resolved_config(base_config, run_dir=run_dir, smoke=True, smoke_evals=123)

    validate_experiment_spec(resolved, allowed_overrides=_allowed_overrides())
    defaults = resolved["defaults"]
    assert defaults["max_evaluations"] == 123
    assert defaults["output_root"] == str(run_dir / "results")


def test_run_plan_writes_report_with_stubbed_execution(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    plan_dir, _ = _materialized_plan(tmp_path)

    def _fake_run(config_path: Path) -> int:
        assert config_path.is_file()
        return 0

    monkeypatch.setattr("vamos.assist.run.run_with_config_path", _fake_run)

    summary = run_plan(
        plan_dir=plan_dir,
        smoke=True,
        smoke_evals=111,
    )

    assert summary["status"] == "ok"
    run_dir = Path(str(summary["run_dir"]))
    resolved_path = run_dir / "resolved_config.json"
    report_path = run_dir / "run_report.json"
    assert resolved_path.is_file()
    assert report_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    for key in (
        "status",
        "exit_code",
        "plan_dir",
        "base_config_path",
        "resolved_config_path",
        "run_dir",
        "started_at",
        "ended_at",
        "smoke",
        "smoke_evals",
        "execution_mode",
        "warnings",
    ):
        assert key in report


def test_run_with_config_path_prefers_in_process(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "resolved_config.json"
    config_path.write_text("{}", encoding="utf-8")

    calls = {"in_process": 0, "subprocess": 0}

    def _fake_in_process(path: str) -> int:
        assert path == str(config_path)
        calls["in_process"] += 1
        return 0

    def _fake_subprocess(path: Path) -> int:
        del path
        calls["subprocess"] += 1
        return 1

    monkeypatch.setattr("vamos.experiment.cli.main.run_from_config_path", _fake_in_process)
    monkeypatch.setattr("vamos.assist.run._run_with_subprocess", _fake_subprocess)

    exit_code = run_with_config_path(config_path)
    assert exit_code == 0
    assert calls == {"in_process": 1, "subprocess": 0}


def test_run_with_config_path_falls_back_to_subprocess(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "resolved_config.json"
    config_path.write_text("{}", encoding="utf-8")

    calls = {"in_process": 0, "subprocess": 0}

    def _fake_in_process(path: str) -> int:
        assert path == str(config_path)
        calls["in_process"] += 1
        raise RuntimeError("forced in-process failure")

    def _fake_subprocess(path: Path) -> int:
        assert path == config_path
        calls["subprocess"] += 1
        return 0

    monkeypatch.setattr("vamos.experiment.cli.main.run_from_config_path", _fake_in_process)
    monkeypatch.setattr("vamos.assist.run._run_with_subprocess", _fake_subprocess)

    exit_code = run_with_config_path(config_path)
    assert exit_code == 0
    assert calls == {"in_process": 1, "subprocess": 1}
