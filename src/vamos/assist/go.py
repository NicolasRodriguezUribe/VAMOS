from __future__ import annotations

from pathlib import Path
from typing import Any

from .apply import apply_plan
from .plan import create_plan
from .providers import MockPlanProvider, OpenAIPlanProvider, PlanProvider
from .run import run_plan


def _coerce_exit_code(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError:
            return 1
    return 1


def _make_provider(mode: str, provider_name: str) -> PlanProvider | None:
    normalized_mode = mode.strip().lower()
    if normalized_mode != "auto":
        return None

    normalized_provider = provider_name.strip().lower()
    if normalized_provider == "mock":
        return MockPlanProvider()
    if normalized_provider == "openai":
        return OpenAIPlanProvider()
    raise ValueError(f"Unknown provider '{provider_name}'. Supported providers: mock, openai.")


def go(
    prompt: str,
    mode: str = "template",
    provider_name: str = "mock",
    template: str | None = None,
    problem_type: str = "real",
    out_dir: Path | None = None,
    overwrite: bool = False,
    smoke: bool = False,
    smoke_evals: int = 200,
) -> dict[str, Any]:
    provider = _make_provider(mode=mode, provider_name=provider_name)
    plan_dir = create_plan(
        prompt=prompt,
        template=template,
        problem_type=problem_type,
        out_dir=out_dir,
        mode=mode,
        provider=provider,
        provider_name=provider_name if mode == "auto" else None,
    )
    project_dir = apply_plan(plan_dir, out_dir=None, overwrite=overwrite)

    project_config = project_dir / "config.json"
    summary: dict[str, Any] = {
        "plan_dir": str(plan_dir),
        "project_dir": str(project_dir),
        "plan_paths": {
            "config": str(plan_dir / "config.json"),
            "catalog": str(plan_dir / "catalog.json"),
            "plan": str(plan_dir / "plan.json"),
            "prompt": str(plan_dir / "prompt.txt"),
        },
        "recommended_commands": [f"vamos --config {project_config}"],
    }

    if smoke:
        run_summary = run_plan(
            plan_dir=plan_dir,
            smoke=True,
            smoke_evals=smoke_evals,
            overwrite=overwrite,
        )
        run_info = {
            "run_dir": str(run_summary.get("run_dir", "")),
            "resolved_config_path": str(run_summary.get("resolved_config_path", "")),
            "run_report_path": str(run_summary.get("run_report_path", "")),
            "exit_code": _coerce_exit_code(run_summary.get("exit_code", 1)),
            "status": str(run_summary.get("status", "error")),
        }
        summary["run"] = run_info
        resolved_cfg = run_info["resolved_config_path"]
        if resolved_cfg:
            summary["recommended_commands"].append(f"vamos --config {resolved_cfg}")

    return summary


__all__ = ["go"]
