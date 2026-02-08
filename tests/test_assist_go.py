from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from vamos.assist.go import go
from vamos.experiment.cli.quickstart import available_templates


def _template_key() -> str:
    templates = available_templates()
    if "demo" in templates:
        return "demo"
    return templates[0]


def test_go_template_mode_creates_plan_and_project(tmp_path: Path) -> None:
    plan_dir = tmp_path / "go_plan_no_run"
    result = go(
        prompt="Create a runnable plan and project.",
        mode="template",
        template=_template_key(),
        out_dir=plan_dir,
    )

    assert result["plan_dir"] == str(plan_dir)
    project_dir = plan_dir / "project"
    assert result["project_dir"] == str(project_dir)
    assert (plan_dir / "config.json").is_file()
    assert (plan_dir / "catalog.json").is_file()
    assert (plan_dir / "plan.json").is_file()
    assert (plan_dir / "prompt.txt").is_file()
    assert (project_dir / "config.json").is_file()
    assert "run" not in result


def test_go_template_mode_with_smoke_uses_stubbed_runner(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    def _fake_run(config_path: Path) -> int:
        assert config_path.is_file()
        return 0

    monkeypatch.setattr("vamos.assist.run.run_with_config_path", _fake_run)

    plan_dir = tmp_path / "go_plan_smoke"
    result = go(
        prompt="Run a smoke pass after planning.",
        mode="template",
        template=_template_key(),
        out_dir=plan_dir,
        smoke=True,
        smoke_evals=111,
    )

    run_info = result.get("run")
    assert isinstance(run_info, dict)
    assert run_info.get("status") == "ok"
    assert run_info.get("exit_code") == 0

    run_dir = Path(str(run_info["run_dir"]))
    assert run_dir.is_dir()
    assert (run_dir / "resolved_config.json").is_file()
    assert (run_dir / "run_report.json").is_file()
