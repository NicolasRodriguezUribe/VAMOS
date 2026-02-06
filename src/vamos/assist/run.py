from __future__ import annotations

import json
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from vamos.engine.config.loader import load_experiment_spec
from vamos.engine.config.spec import validate_experiment_spec
from vamos.experiment.cli.args import build_parser, build_pre_parser
from vamos.experiment.cli.loaders import load_spec_defaults
from vamos.experiment.cli.spec_args import parser_spec_keys
from vamos.foundation.core.experiment_config import ExperimentConfig

_EVAL_KEYS: tuple[str, ...] = ("max_evaluations", "max_evals", "evaluations")
_GEN_KEYS: tuple[str, ...] = ("max_generations", "generations")
_LAST_EXECUTION_MODE: str = "in_process"


@lru_cache(maxsize=1)
def _spec_allowed_overrides() -> tuple[str, ...]:
    default_config = ExperimentConfig()
    pre_parser = build_pre_parser()
    spec_defaults = load_spec_defaults(None)
    parser = build_parser(default_config=default_config, pre_parser=pre_parser, spec_defaults=spec_defaults)
    return tuple(sorted(parser_spec_keys(parser)))


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_mapping(value: object, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a JSON/YAML object.")
    return value


def _validate_spec(config_data: object) -> None:
    try:
        validate_experiment_spec(config_data, allowed_overrides=_spec_allowed_overrides())
    except Exception as exc:
        raise ValueError(f"Invalid config: {exc}") from exc


def select_config_path(plan_dir: Path, config_override: Path | None = None) -> Path:
    if config_override is not None:
        candidate = Path(config_override)
        if not candidate.is_file():
            raise ValueError(f"Config file not found: {candidate}")
        return candidate

    project_cfg = plan_dir / "project" / "config.json"
    if project_cfg.is_file():
        return project_cfg
    plan_cfg = plan_dir / "config.json"
    if plan_cfg.is_file():
        return plan_cfg
    raise ValueError("No config found in plan directory.")


def prepare_run_dir(plan_dir: Path, out_dir: Path | None, smoke: bool, overwrite: bool) -> Path:
    if out_dir is not None:
        run_dir = Path(out_dir)
    else:
        suffix = "_smoke" if smoke else ""
        run_dir = plan_dir / "runs" / f"run_{_timestamp()}{suffix}"

    if run_dir.exists():
        if not overwrite:
            raise ValueError(f"Run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _override_smoke_budget(config: dict[str, Any], smoke_evals: int) -> str:
    defaults = config.get("defaults")
    if isinstance(defaults, dict):
        for key in _EVAL_KEYS:
            if key in defaults:
                defaults[key] = int(smoke_evals)
                return f"defaults.{key}"
        for key in _GEN_KEYS:
            if key in defaults:
                defaults[key] = int(smoke_evals)
                return f"defaults.{key}"
    raise ValueError(
        "Smoke override is not supported for this config: no stopping budget key found in defaults "
        "(expected one of max_evaluations/max_evals/evaluations/max_generations/generations)."
    )


def make_resolved_config(base_config: dict[str, Any], run_dir: Path, smoke: bool, smoke_evals: int) -> dict[str, Any]:
    resolved = deepcopy(base_config)
    defaults = resolved.get("defaults")
    if defaults is None:
        defaults = {}
        resolved["defaults"] = defaults
    defaults_map = _require_mapping(defaults, path="defaults")
    defaults_map["output_root"] = str(run_dir / "results")

    if smoke:
        _override_smoke_budget(resolved, smoke_evals)
    return resolved


def _run_with_subprocess(config_path: Path) -> int:
    cmd = [sys.executable, "-m", "vamos.experiment.cli.main", "--config", str(config_path)]
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def _run_with_config_path_detailed(config_path: Path) -> tuple[int, str]:
    from vamos.experiment.cli.main import run_from_config_path

    try:
        return int(run_from_config_path(str(config_path))), "in_process"
    except Exception:
        return _run_with_subprocess(config_path), "subprocess"


def run_with_config_path(config_path: Path) -> int:
    global _LAST_EXECUTION_MODE
    exit_code, mode = _run_with_config_path_detailed(config_path)
    _LAST_EXECUTION_MODE = mode
    return int(exit_code)


def run_plan(
    plan_dir: Path,
    config_override: Path | None = None,
    out_dir: Path | None = None,
    smoke: bool = False,
    smoke_evals: int = 200,
    overwrite: bool = False,
) -> dict[str, object]:
    resolved_plan_dir = Path(plan_dir)
    if not resolved_plan_dir.is_dir():
        raise ValueError(f"Plan directory not found: {resolved_plan_dir}")
    if smoke_evals <= 0:
        raise ValueError("smoke_evals must be a positive integer.")

    base_config_path = select_config_path(resolved_plan_dir, config_override=config_override)
    base_config = load_experiment_spec(str(base_config_path))
    _validate_spec(base_config)

    run_dir = prepare_run_dir(resolved_plan_dir, out_dir=out_dir, smoke=smoke, overwrite=overwrite)

    warnings: list[str] = []
    warnings.append("Archive output path is controlled via defaults.output_root in this schema.")

    resolved_config = make_resolved_config(base_config, run_dir=run_dir, smoke=smoke, smoke_evals=smoke_evals)
    _validate_spec(resolved_config)

    resolved_config_path = run_dir / "resolved_config.json"
    resolved_config_path.write_text(json.dumps(resolved_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    command = [sys.executable, "-m", "vamos.experiment.cli.main", "--config", str(resolved_config_path)]
    started_at = _iso_now()
    status = "ok"
    error_message: str | None = None
    execution_mode = "in_process"
    try:
        exit_code = run_with_config_path(resolved_config_path)
        execution_mode = _LAST_EXECUTION_MODE
    except Exception as exc:  # pragma: no cover - defensive
        exit_code = 1
        status = "error"
        error_message = str(exc)
        warnings.append(f"Execution failed: {exc}")
    else:
        if exit_code != 0:
            status = "error"
    ended_at = _iso_now()

    run_report: dict[str, object] = {
        "status": status,
        "exit_code": int(exit_code),
        "plan_dir": str(resolved_plan_dir),
        "base_config_path": str(base_config_path),
        "resolved_config_path": str(resolved_config_path),
        "run_dir": str(run_dir),
        "started_at": started_at,
        "ended_at": ended_at,
        "smoke": bool(smoke),
        "smoke_evals": int(smoke_evals),
        "execution_mode": execution_mode,
        "command": command,
        "warnings": warnings,
    }
    if error_message is not None:
        run_report["error_message"] = error_message

    run_report_path = run_dir / "run_report.json"
    run_report_path.write_text(json.dumps(run_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_report["run_report_path"] = str(run_report_path)
    return run_report


__all__ = [
    "make_resolved_config",
    "prepare_run_dir",
    "run_plan",
    "run_with_config_path",
    "select_config_path",
]
