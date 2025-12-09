"""
Helpers for building run metadata and operator summaries.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Optional

from vamos.version import __version__


def git_revision(project_root: Path) -> Optional[str]:
    """
    Return current git commit hash if available, otherwise None.
    Safe to call in packaged installations without git.
    """
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return rev.decode().strip() or None


def serialize_operator_tuple(op_tuple):
    if not op_tuple:
        return None
    name, params = op_tuple
    return {"name": name, "params": params}


def collect_operator_metadata(cfg_data) -> dict:
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
    selection,
    algorithm_name: str,
    engine_name: str,
    cfg_data,
    metrics: dict,
    *,
    kernel_backend,
    seed: int,
    config,
    project_root: Path,
) -> dict:
    """
    Assemble the metadata payload for a single run.
    """
    timestamp = datetime.now(UTC).isoformat()
    problem = selection.instantiate()
    problem_info = {
        "label": selection.spec.label,
        "key": selection.spec.key,
        "n_var": selection.n_var,
        "n_obj": selection.n_obj,
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
        "vamos_version": __version__,
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
