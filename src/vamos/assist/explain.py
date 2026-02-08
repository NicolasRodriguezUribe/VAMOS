from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .run import select_config_path

_BUDGET_KEYS: tuple[str, ...] = ("max_evaluations", "max_evals", "evaluations", "max_generations", "generations")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in: {path}")
    return raw


def _extract_defaults(config: dict[str, Any]) -> dict[str, Any]:
    defaults = config.get("defaults")
    if not isinstance(defaults, dict):
        return {}

    summary: dict[str, Any] = {}
    if "algorithm" in defaults:
        summary["algorithm"] = defaults["algorithm"]
    if "engine" in defaults:
        summary["kernel"] = defaults["engine"]
    for key in _BUDGET_KEYS:
        if key in defaults:
            summary["stopping_budget"] = {"key": key, "value": defaults[key]}
            break
    if "output_root" in defaults:
        summary["output_root"] = defaults["output_root"]
    return summary


def _diff_defaults(base_config: dict[str, Any], resolved_config: dict[str, Any]) -> dict[str, Any]:
    base_defaults = base_config.get("defaults")
    resolved_defaults = resolved_config.get("defaults")
    if not isinstance(base_defaults, dict) or not isinstance(resolved_defaults, dict):
        return {}

    changes: dict[str, Any] = {}
    for key in sorted(set(base_defaults) | set(resolved_defaults)):
        before = base_defaults.get(key)
        after = resolved_defaults.get(key)
        if before != after:
            changes[key] = {"before": before, "after": after}
    return changes


def summarize_plan(plan_dir: Path, run_dir: Path | None = None) -> dict[str, object]:
    resolved_plan_dir = Path(plan_dir)
    if not resolved_plan_dir.is_dir():
        raise ValueError(f"Plan directory not found: {resolved_plan_dir}")

    plan_metadata_path = resolved_plan_dir / "plan.json"
    plan_config_path = resolved_plan_dir / "config.json"
    metadata = _load_json(plan_metadata_path)
    _load_json(plan_config_path)  # ensure canonical plan config exists

    execution_config_path = select_config_path(resolved_plan_dir)
    execution_config = _load_json(execution_config_path)

    paths: dict[str, str] = {
        "plan_dir": str(resolved_plan_dir),
        "plan_config_path": str(plan_config_path),
        "plan_metadata_path": str(plan_metadata_path),
    }
    prompt_path = resolved_plan_dir / "prompt.txt"
    if prompt_path.is_file():
        paths["prompt_path"] = str(prompt_path)
    catalog_path = resolved_plan_dir / "catalog.json"
    if catalog_path.is_file():
        paths["catalog_path"] = str(catalog_path)
    project_dir = resolved_plan_dir / "project"
    if project_dir.is_dir():
        paths["project_dir"] = str(project_dir)
    if execution_config_path != plan_config_path:
        paths["execution_config_path"] = str(execution_config_path)

    summary: dict[str, object] = {
        "template": metadata.get("template"),
        "problem_type": metadata.get("problem_type"),
        "defaults": _extract_defaults(execution_config),
        "paths": paths,
        "recommended_next_commands": [
            f"vamos --config {execution_config_path}",
            f"python -m vamos.experiment.cli.main --config {execution_config_path}",
        ],
    }

    if run_dir is not None:
        resolved_run_dir = Path(run_dir)
        run_report_path = resolved_run_dir / "run_report.json"
        resolved_config_path = resolved_run_dir / "resolved_config.json"
        run_section: dict[str, object] = {
            "run_dir": str(resolved_run_dir),
            "run_report_path": str(run_report_path),
            "resolved_config_path": str(resolved_config_path),
        }

        if run_report_path.is_file():
            report = _load_json(run_report_path)
            for key in ("status", "exit_code", "started_at", "ended_at", "smoke", "smoke_evals", "warnings"):
                if key in report:
                    run_section[key] = report[key]
        if resolved_config_path.is_file():
            resolved_config = _load_json(resolved_config_path)
            run_section["resolved_overrides"] = {
                "defaults": _diff_defaults(execution_config, resolved_config),
            }
        summary["run"] = run_section
        commands = summary["recommended_next_commands"]
        if isinstance(commands, list):
            commands.append(f"vamos --config {resolved_config_path}")

    return summary


__all__ = ["summarize_plan"]
