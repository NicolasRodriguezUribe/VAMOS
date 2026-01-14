"""
Helpers to discover and load experiment results following the standard layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

import numpy as np

from vamos.foundation.core.io_utils import RESULT_FILES

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None


@dataclass(frozen=True)
class RunInfo:
    path: Path
    metadata_path: Path
    problem: str
    algorithm: str
    engine: str
    seed: int | None
    study: str | None


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


def _coerce_array(arr: np.ndarray | float | int) -> np.ndarray:
    if isinstance(arr, (float, int)):
        return np.array([[arr]])
    if arr.ndim == 1:
        return arr.reshape(-1, arr.shape[0] if arr.size > 1 else 1)
    return arr


def _try_load_csv(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        data = np.loadtxt(path, delimiter=",")
    except Exception:
        return None
    return _coerce_array(np.asarray(data))


def discover_runs(base_dir: str | Path = "results") -> list[RunInfo]:
    """
    Recursively locate run directories by scanning for metadata.json files.
    """
    base = Path(base_dir)
    runs: list[RunInfo] = []
    for meta_path in base.rglob(RESULT_FILES["metadata"]):
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        problem = metadata.get("problem", {}) or {}
        problem_key = problem.get("key") or problem.get("label") or meta_path.parent.parent.parent.name
        algorithm = metadata.get("algorithm") or meta_path.parent.parent.name
        engine = metadata.get("backend") or meta_path.parent.name
        seed = metadata.get("seed")
        if seed is None:
            try:
                seed = int(str(meta_path.parent.name).split("_")[-1])
            except Exception:
                seed = None
        study = metadata.get("title")
        runs.append(
            RunInfo(
                path=meta_path.parent,
                metadata_path=meta_path,
                problem=str(problem_key),
                algorithm=str(algorithm),
                engine=str(engine),
                seed=seed,
                study=study,
            )
        )
    return sorted(runs, key=lambda r: (r.problem, r.algorithm, r.engine, r.seed or -1))


def load_run_data(run: RunInfo) -> RunData:
    metadata = json.loads(run.metadata_path.read_text(encoding="utf-8"))
    artifacts = metadata.get("artifacts", {})
    run_dir = run.path

    def resolve(name: str) -> Path | None:
        filename = artifacts.get(name) or RESULT_FILES.get(name)
        return run_dir / filename if filename else None

    fun_path = resolve("fun")
    x_path = resolve("x")
    g_path = resolve("g")
    archive_fun_path = resolve("archive_fun")
    archive_x_path = resolve("archive_x")
    archive_g_path = resolve("archive_g")

    F = _try_load_csv(fun_path) if fun_path else None
    X = _try_load_csv(x_path) if x_path else None
    G = _try_load_csv(g_path) if g_path else None
    archive_F = _try_load_csv(archive_fun_path) if archive_fun_path else None
    archive_X = _try_load_csv(archive_x_path) if archive_x_path else None
    archive_G = _try_load_csv(archive_g_path) if archive_g_path else None

    return RunData(
        info=run,
        F=F,
        X=X,
        G=G,
        archive_F=archive_F,
        archive_X=archive_X,
        archive_G=archive_G,
        metadata=metadata,
    )


def aggregate_results(runs: Iterable[RunInfo]) -> object:
    """
    Aggregate metadata/metrics for a collection of runs.
    Returns a pandas DataFrame if pandas is installed, otherwise a list of dicts.
    """
    records = []
    for run in runs:
        meta = json.loads(run.metadata_path.read_text(encoding="utf-8"))
        metrics = meta.get("metrics", {}) or {}
        record = {
            "problem": run.problem,
            "algorithm": run.algorithm,
            "engine": run.engine,
            "seed": run.seed,
            "study": run.study,
            **metrics,
        }
        records.append(record)
    if pd:
        return pd.DataFrame.from_records(records)
    return records


__all__ = ["RunInfo", "RunData", "discover_runs", "load_run_data", "aggregate_results"]
