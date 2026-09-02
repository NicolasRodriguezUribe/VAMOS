from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vamos import load_result, load_run
from vamos.experiment.artifacts import RunManifest
from vamos.experiment.optimization_result import OptimizationResult


@dataclass(frozen=True)
class CanonicalRun:
    root: Path
    manifest: RunManifest
    result: OptimizationResult


def discover_run_paths(root: Path) -> list[Path]:
    """Discover canonical runs only through their manifest."""
    return sorted(path.parent for path in root.rglob("manifest.json") if path.is_file())


def load_canonical_run(path: Path) -> CanonicalRun:
    stored = load_run(path, verify="required")
    if stored.manifest.artifact("result_bundle") is None:
        raise ValueError(f"Canonical run at {path} has no numerical result bundle.")
    result = load_result(path, verify="required")
    return CanonicalRun(root=stored.root, manifest=stored.manifest, result=result)


def result_runs(root: Path) -> list[CanonicalRun]:
    """Load canonical runs that declare a numerical result bundle."""
    runs: list[CanonicalRun] = []
    for path in discover_run_paths(root):
        stored = load_run(path, verify="required")
        if stored.manifest.artifact("result_bundle") is None:
            continue
        runs.append(CanonicalRun(root=stored.root, manifest=stored.manifest, result=load_result(path, verify="required")))
    return runs


def component_name(value: object) -> str:
    component = _mapping(value)
    resolution = _mapping(component.get("resolution"))
    resolved_name = resolution.get("name")
    if isinstance(resolved_name, str):
        return resolved_name
    component_id = component.get("component_id")
    if isinstance(component_id, str) and ":" in component_id:
        return component_id.split(":", 1)[1].split("@", 1)[0]
    return "unknown"


def canonical_run_key(run: CanonicalRun) -> tuple[str, str, str, int]:
    resolved = run.manifest.resolved_spec
    backend = _mapping(resolved.get("backend"))
    return (
        component_name(resolved.get("algorithm")),
        component_name(resolved.get("problem")),
        component_name(backend.get("kernel")),
        int(resolved["seed"]),
    )


def run_rows(root: Path, *, campaign: str) -> list[dict[str, Any]]:
    return [run_row(run, campaign=campaign) for run in result_runs(root)]


def run_row(run: CanonicalRun, *, campaign: str) -> dict[str, Any]:
    resolved = run.manifest.resolved_spec
    algorithm = _mapping(resolved.get("algorithm"))
    problem = _mapping(resolved.get("problem"))
    backend = _mapping(resolved.get("backend"))
    kernel = _mapping(backend.get("kernel"))
    termination = _mapping(resolved.get("termination"))
    outcome = _mapping(run.manifest.get("outcome"))
    provenance = _mapping(run.manifest.get("provenance"))
    implementation = _mapping(provenance.get("implementation"))
    source = _mapping(provenance.get("source"))
    timestamps = _mapping(provenance.get("timestamps"))
    population = _mapping(resolved.get("population"))
    problem_config = _mapping(problem.get("config"))
    algorithm_config = _mapping(algorithm.get("config"))
    metrics = _mapping(outcome.get("metrics"))
    labels = _mapping(run.manifest.get("labels"))

    problem_name = component_name(problem)
    row: dict[str, Any] = {
        "run_path": run.root.as_posix(),
        "run_id": run.manifest.run_id,
        "task_id": run.manifest.task_id,
        "status": run.manifest.status,
        "campaign": campaign,
        "variant": labels.get("variant"),
        "suite": infer_suite_from_problem(problem_name),
        "algorithm": component_name(algorithm),
        "engine": component_name(kernel),
        "problem": problem_name,
        "seed": resolved.get("seed"),
        "max_evaluations": _mapping(termination.get("config")).get("max_evaluations"),
        "population_size": population.get("initial_size"),
        "n_obj": problem_config.get("n_obj"),
        "n_var": problem_config.get("n_var"),
        "runtime_seconds": _runtime_seconds(outcome.get("runtime_ms")),
        "front_size": _rows(run.result.F),
        "objective_count": _columns(run.result.F),
        "decision_rows": _rows(run.result.X),
        "decision_columns": _columns(run.result.X),
        "git_revision": source.get("git_sha"),
        "timestamp": timestamps.get("completed_at"),
        "vamos_version": implementation.get("vamos_version"),
        "config_keys": ",".join(sorted(str(key) for key in algorithm_config)),
    }
    if run.result.F is not None:
        values = np.asarray(run.result.F)
        for index in range(values.shape[1]):
            row[f"obj{index}_min"] = float(np.min(values[:, index]))
            row[f"obj{index}_max"] = float(np.max(values[:, index]))
    flatten_mapping("metrics", metrics, row)
    return row


def write_tidy_csv(path: Path, rows: list[dict[str, Any]], *, core_columns: list[str]) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = {key for row in rows for key in row}
    columns = core_columns + sorted(keys - set(core_columns))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return columns


def flatten_mapping(prefix: str, value: Mapping[str, Any], output: dict[str, Any]) -> None:
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flatten_mapping(name, item, output)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            output[name] = item


def infer_suite_from_problem(problem_key: str) -> str:
    match = re.match(r"^([a-z]+)", problem_key.lower())
    return (match.group(1) if match else "unknown").upper()


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _runtime_seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) / 1000.0


def _rows(value: np.ndarray[Any, Any] | None) -> int | None:
    return int(value.shape[0]) if value is not None else None


def _columns(value: np.ndarray[Any, Any] | None) -> int | None:
    return int(value.shape[1]) if value is not None and value.ndim > 1 else None


__all__ = [
    "CanonicalRun",
    "canonical_run_key",
    "component_name",
    "discover_run_paths",
    "flatten_mapping",
    "infer_suite_from_problem",
    "load_canonical_run",
    "result_runs",
    "run_row",
    "run_rows",
    "write_tidy_csv",
]
