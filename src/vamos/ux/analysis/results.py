"""Discover and analyze canonical v1 run directories."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vamos.run_artifacts import load_result, load_run

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None


@dataclass(frozen=True)
class RunInfo:
    path: Path
    manifest_path: Path
    problem: str
    algorithm: str
    engine: str
    seed: int
    study: str | None
    completed_at: str | None


@dataclass
class RunData:
    info: RunInfo
    F: np.ndarray | None
    X: np.ndarray | None
    G: np.ndarray | None
    archive_F: np.ndarray | None
    archive_X: np.ndarray | None
    archive_G: np.ndarray | None
    metadata: dict[str, object]


def discover_runs(base_dir: str | Path = "results") -> list[RunInfo]:
    """Return valid canonical runs found beneath *base_dir*."""
    runs: list[RunInfo] = []
    for manifest_path in Path(base_dir).rglob("manifest.json"):
        try:
            manifest = load_run(manifest_path.parent, verify="manifest").manifest
            resolved = manifest.resolved_spec
            timestamps = manifest.get("timestamps")
            labels = manifest.get("labels")
            seed = resolved.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                continue
            runs.append(
                RunInfo(
                    path=manifest_path.parent,
                    manifest_path=manifest_path,
                    problem=_component_name(resolved.get("problem")),
                    algorithm=_component_name(resolved.get("algorithm")),
                    engine=_kernel_name(resolved.get("backend")),
                    seed=seed,
                    study=str(labels.get("study")) if isinstance(labels, Mapping) and labels.get("study") else None,
                    completed_at=(
                        str(timestamps.get("completed_at")) if isinstance(timestamps, Mapping) and timestamps.get("completed_at") else None
                    ),
                )
            )
        except Exception:
            continue
    return sorted(runs, key=lambda run: (run.problem, run.algorithm, run.engine, run.seed))


def load_run_data(run: RunInfo) -> RunData:
    """Load arrays only through the canonical data-only reader."""
    stored = load_run(run.path)
    result = load_result(run.path)
    archive = result.data.get("archive")
    archive_data = archive if isinstance(archive, Mapping) else {}
    return RunData(
        info=run,
        F=result.F,
        X=result.X,
        G=_array_or_none(result.data.get("G")),
        archive_F=_array_or_none(archive_data.get("F")),
        archive_X=_array_or_none(archive_data.get("X")),
        archive_G=_array_or_none(archive_data.get("G")),
        metadata=stored.manifest.as_dict(),
    )


def aggregate_results(runs: Iterable[RunInfo]) -> object:
    """Aggregate manifest outcome fields, returning a DataFrame when available."""
    records: list[dict[str, object]] = []
    for run in runs:
        manifest = load_run(run.path, verify="manifest").manifest
        outcome = manifest.get("outcome")
        outcome_data = outcome if isinstance(outcome, Mapping) else {}
        metrics = outcome_data.get("metrics")
        metric_data = dict(metrics) if isinstance(metrics, Mapping) else {}
        records.append(
            {
                "problem": run.problem,
                "algorithm": run.algorithm,
                "engine": run.engine,
                "seed": run.seed,
                "study": run.study,
                "evaluations": outcome_data.get("evaluations"),
                "generations": outcome_data.get("generations"),
                "runtime_ms": outcome_data.get("runtime_ms"),
                "termination": outcome_data.get("termination_reason"),
                **metric_data,
            }
        )
    if pd is not None:
        return pd.DataFrame.from_records(records)
    return records


def _array_or_none(value: object) -> np.ndarray | None:
    return value if isinstance(value, np.ndarray) else None


def _component_name(value: object) -> str:
    if isinstance(value, Mapping):
        resolution = value.get("resolution")
        if isinstance(resolution, Mapping) and isinstance(resolution.get("name"), str):
            return str(resolution["name"])
        component_id = value.get("component_id")
        if isinstance(component_id, str) and ":" in component_id:
            return component_id.split(":", 1)[1].split("@", 1)[0]
    return "unknown"


def _kernel_name(value: object) -> str:
    kernel = value.get("kernel") if isinstance(value, Mapping) else None
    return _component_name(kernel)


__all__ = ["RunInfo", "RunData", "discover_runs", "load_run_data", "aggregate_results"]
