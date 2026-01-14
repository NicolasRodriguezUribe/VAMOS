from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Iterable

import numpy as np


@dataclass
class RunRecord:
    suite_name: str | None
    experiment_id: str
    problem_name: str
    algorithm_name: str
    seed: int
    fun: np.ndarray
    var: np.ndarray | None
    archive_fun: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrontRecord:
    problem_name: str
    algorithm_name: str
    points_F: np.ndarray
    points_X: np.ndarray | None
    constraints: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _load_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_run_from_directory(run_dir: Path) -> RunRecord:
    """
    Load a single run (FUN/VAR/metadata) produced by run_single/StudyRunner.
    """
    run_dir = run_dir.resolve()
    fun_path = run_dir / "FUN.csv"
    if not fun_path.exists():
        raise FileNotFoundError(f"Missing FUN.csv in {run_dir}")
    fun = _load_csv(fun_path)
    var_path = run_dir / "VAR.csv"
    var = _load_csv(var_path) if var_path.exists() else None
    archive_path = run_dir / "ARCHIVE_FUN.csv"
    archive_fun = _load_csv(archive_path) if archive_path.exists() else None
    metadata_path = run_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    config_path = run_dir / "resolved_config.json"
    if config_path.exists():
        try:
            metadata["config"] = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Heuristics to determine problem/algorithm/seed when metadata is absent
    problem_name = metadata.get("problem", {}).get("key") or metadata.get("problem_key")
    algorithm_name = metadata.get("algorithm")
    seed = metadata.get("seed", -1)
    # experiment_id = run_dir.name  # Available for future use
    parts = list(run_dir.parts)
    if problem_name is None and len(parts) >= 3:
        problem_name = parts[-3]
    if algorithm_name is None and len(parts) >= 2:
        algorithm_name = parts[-2]
    if seed == -1:
        try:
            seed = int(parts[-1].split("_")[-1])
        except Exception:
            seed = -1
    suite_name = metadata.get("suite") or None
    return RunRecord(
        suite_name=suite_name,
        experiment_id=str(run_dir),
        problem_name=problem_name or "unknown_problem",
        algorithm_name=algorithm_name or "unknown_algorithm",
        seed=seed,
        fun=fun,
        var=var,
        archive_fun=archive_fun,
        metadata=metadata,
    )


def _iter_run_dirs(study_dir: Path) -> Iterable[Path]:
    # Expect structure results/PROBLEM/ALGO/ENGINE/seed_x
    for fun_path in study_dir.rglob("FUN.csv"):
        yield fun_path.parent


def load_runs_from_study(study_dir: Path) -> list[RunRecord]:
    """
    Load all run directories underneath a study root.
    """
    runs: list[RunRecord] = []
    for run_dir in _iter_run_dirs(study_dir):
        try:
            runs.append(load_run_from_directory(run_dir))
        except FileNotFoundError:
            continue
    return runs


def build_fronts(
    runs: Iterable[RunRecord],
    *,
    problem_filter: str | None = None,
    merge_seeds: bool = True,
) -> list[FrontRecord]:
    """
    Build per-(problem, algorithm) fronts by optionally merging seeds.
    """
    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for run in runs:
        if problem_filter and run.problem_name != problem_filter:
            continue
        key = (
            (run.problem_name, run.algorithm_name)
            if merge_seeds
            else (
                f"{run.problem_name}_seed{run.seed}",
                run.algorithm_name,
            )
        )
        grouped.setdefault(key, []).append(run)

    fronts: list[FrontRecord] = []
    for (problem, algorithm), records in grouped.items():
        F = np.vstack([r.fun for r in records if r.fun.size])
        X = None
        if any(r.var is not None for r in records):
            X = np.vstack([r.var for r in records if r.var is not None])
        constraints = None
        fronts.append(
            FrontRecord(
                problem_name=problem,
                algorithm_name=algorithm,
                points_F=F,
                points_X=X,
                constraints=constraints,
                extra={"seeds": [r.seed for r in records], "config": records[0].metadata.get("config") if records else None},
            )
        )
    return fronts


def normalize_objectives(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    mins = F.min(axis=0)
    maxs = F.max(axis=0)
    span = np.where(maxs - mins == 0, 1.0, maxs - mins)
    return np.asarray((F - mins) / span, dtype=float)
