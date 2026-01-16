"""
Helpers for building run metadata and operator summaries.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vamos.foundation.version import get_version


def git_revision(project_root: Path) -> str | None:
    """
    Return current git commit hash if available, otherwise None.
    Safe to call in packaged installations without git.
    """
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return rev.decode().strip() or None


def serialize_operator_tuple(op_tuple: object) -> dict[str, Any] | None:
    if not op_tuple:
        return None
    if not isinstance(op_tuple, tuple) or len(op_tuple) != 2:
        return None
    name, params = op_tuple
    return {"name": str(name), "params": params}


def collect_operator_metadata(cfg_data: Any) -> dict[str, Any]:
    if cfg_data is None:
        return {}
    payload = {}
    for key in ("crossover", "mutation", "repair"):
        value = getattr(cfg_data, key, None)
        formatted = serialize_operator_tuple(value)
        if formatted:
            payload[key] = formatted
    return payload


def build_run_metadata(
    selection: Any,
    algorithm_name: str,
    engine_name: str,
    cfg_data: Any,
    metrics: dict[str, Any],
    *,
    kernel_backend: Any,
    seed: int | None,
    config: Any,
    project_root: Path,
) -> dict[str, Any]:
    """
    Assemble the metadata payload for a single run.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    problem = selection.instantiate()
    spec = getattr(selection, "spec", None)
    label = getattr(spec, "label", None) or getattr(spec, "key", "unknown")
    key = getattr(spec, "key", "unknown")
    problem_info = {
        "label": label,
        "key": key,
        "n_var": getattr(selection, "n_var", None),
        "n_obj": getattr(selection, "n_obj", None),
        "encoding": getattr(problem, "encoding", "continuous"),
    }
    try:
        problem_info["description"] = selection.spec.description
    except Exception:
        pass

    kernel_caps = sorted(set(kernel_backend.capabilities())) if kernel_backend else []
    kernel_info = {
        "name": kernel_backend.__class__.__name__ if kernel_backend else "external",
        "device": kernel_backend.device() if kernel_backend else "external",
        "capabilities": kernel_caps,
    }
    operator_payload = collect_operator_metadata(cfg_data)
    config_payload = cfg_data.to_dict() if hasattr(cfg_data, "to_dict") else None
    metric_payload = {
        "time_ms": metrics["time_ms"],
        "evaluations": metrics["evaluations"],
        "evals_per_sec": metrics["evals_per_sec"],
        "spread": metrics["spread"],
        "termination": metrics.get("termination"),
    }
    if metrics.get("hv_threshold_fraction") is not None:
        metric_payload["hv_threshold_fraction"] = metrics.get("hv_threshold_fraction")
        metric_payload["hv_reference_point"] = metrics.get("hv_reference_point")
        metric_payload["hv_reference_front"] = metrics.get("hv_reference_front")
    metadata = {
        "title": config.title,
        "timestamp": timestamp,
        "algorithm": algorithm_name,
        "backend": engine_name,
        "backend_info": kernel_info,
        "seed": seed,
        "population_size": config.population_size,
        "max_evaluations": config.max_evaluations,
        "vamos_version": get_version(),
        "git_revision": git_revision(project_root),
        "problem": problem_info,
        "config": config_payload,
        "metrics": metric_payload,
    }
    if operator_payload:
        metadata["operators"] = operator_payload
    return metadata


__all__ = [
    "git_revision",
    "serialize_operator_tuple",
    "collect_operator_metadata",
    "build_run_metadata",
]
