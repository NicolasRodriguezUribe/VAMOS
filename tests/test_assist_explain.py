from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from vamos.assist.apply import apply_plan
from vamos.assist.explain import summarize_plan
from vamos.assist.plan import create_plan
from vamos.assist.run import run_plan


def test_summarize_plan_with_run_is_json_serializable(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    plan_dir = create_plan(
        prompt="Explain assist artifacts.",
        template="demo",
        problem_type="real",
        out_dir=tmp_path / "plan",
    )
    apply_plan(plan_dir)

    def _fake_run(config_path: Path) -> int:
        assert config_path.is_file()
        return 0

    monkeypatch.setattr("vamos.assist.run.run_with_config_path", _fake_run)
    run_summary = run_plan(plan_dir=plan_dir, smoke=True, smoke_evals=77)
    run_dir = Path(str(run_summary["run_dir"]))

    summary = summarize_plan(plan_dir, run_dir=run_dir)
    json.dumps(summary, sort_keys=True)

    assert "template" in summary
    assert "defaults" in summary
    assert "paths" in summary
    assert "recommended_next_commands" in summary
    assert "run" in summary

    run_info = summary["run"]
    assert isinstance(run_info, dict)
    assert run_info.get("status") == "ok"
    overrides = run_info.get("resolved_overrides")
    assert isinstance(overrides, dict)
    defaults_overrides = overrides.get("defaults")
    assert isinstance(defaults_overrides, dict)
    assert "max_evaluations" in defaults_overrides
    assert "output_root" in defaults_overrides
