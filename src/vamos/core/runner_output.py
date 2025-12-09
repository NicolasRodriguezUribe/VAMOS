"""
Output/banners/artifact wiring for runner.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vamos.core.io_utils import write_population, write_metadata, write_timing, ensure_dir
from vamos.core.metadata import build_run_metadata


def print_run_banner(problem, problem_selection, algorithm_label: str, backend_label: str, config) -> None:
    spec = getattr(problem_selection, "spec", None)
    label = getattr(spec, "label", None) or getattr(spec, "key", "unknown")
    description = getattr(spec, "description", None) or ""
    print("=" * 80)
    print(config.title)
    print("=" * 80)
    print(f"Problem: {label}")
    if description:
        print(f"Description: {description}")
    print(f"Decision variables: {problem.n_var}")
    print(f"Objectives: {problem.n_obj}")
    encoding = getattr(problem, "encoding", None)
    if encoding is None:
        encoding = getattr(spec, "encoding", None)
    if encoding is None:
        encoding = "continuous"
    if encoding:
        print(f"Encoding: {encoding}")
    print(f"Algorithm: {algorithm_label}")
    print(f"Backend: {backend_label}")
    print(f"Population size: {config.population_size}")
    print(f"Offspring population size: {config.offspring_size()}")
    print(f"Max evaluations: {config.max_evaluations}")
    print("-" * 80)


def build_metrics(
    algorithm_name: str,
    engine_name: str,
    total_time_ms: float,
    evaluations: int,
    F: np.ndarray,
) -> dict:
    spread = None
    if F.size and F.shape[1] >= 1:
        spread = np.ptp(F[:, 0])
    evals_per_sec = evaluations / max(1e-9, total_time_ms / 1000.0)
    return {
        "algorithm": algorithm_name,
        "engine": engine_name,
        "time_ms": total_time_ms,
        "evaluations": evaluations,
        "evals_per_sec": evals_per_sec,
        "spread": spread,
        "F": F,
    }


def print_run_results(metrics: dict) -> None:
    algo = metrics["algorithm"]
    time_ms = metrics["time_ms"]
    evals = metrics["evaluations"]
    hv_info = ""
    hv = metrics.get("hv")
    if hv is not None:
        hv_info = f" | HV: {hv:.6f}"
    print(f"{algo} -> Time: {time_ms:.2f} ms | Eval/s: {metrics['evals_per_sec']:.1f}{hv_info}")
    spread = metrics.get("spread")
    if spread is not None:
        print(f"Objective 1 spread: {spread:.6f}")


def persist_run_outputs(
    *,
    output_dir: str,
    selection,
    algorithm_name: str,
    engine_name: str,
    cfg_data,
    metrics: dict,
    payload: dict,
    total_time_ms: float,
    hv_stop_config: dict | None,
    config_source: str | None,
    selection_pressure: int,
    external_archive_size: int | None,
    encoding: str,
    problem_override: dict | None,
    autodiff_info: dict | None,
    config,
    kernel_backend: Any,
    project_root: Path,
    nsgaii_variation: dict | None = None,
    moead_variation: dict | None = None,
    smsemoa_variation: dict | None = None,
    nsga3_variation: dict | None = None,
) -> tuple[dict, dict, dict]:
    artifacts = write_population(
        output_dir,
        payload.get("F"),
        payload.get("archive"),
        X=payload.get("X"),
        G=payload.get("G"),
    )
    if payload.get("genealogy"):
        genealogy_path = Path(output_dir) / "genealogy.json"
        genealogy_path.write_text(json.dumps(payload["genealogy"], indent=2), encoding="utf-8")
        artifacts["genealogy"] = genealogy_path.name
    if autodiff_info is not None:
        autodiff_path = Path(output_dir) / "autodiff_constraints.json"
        autodiff_path.write_text(json.dumps(autodiff_info, indent=2), encoding="utf-8")
        artifacts["autodiff_constraints"] = autodiff_path.name
    if payload.get("archive") is not None:
        metrics["archive"] = payload["archive"]

    write_timing(output_dir, total_time_ms)
    metadata = build_run_metadata(
        selection,
        algorithm_name,
        engine_name,
        cfg_data,
        metrics,
        kernel_backend=kernel_backend,
        seed=config.seed,
        config=config,
        project_root=project_root,
    )
    metadata["config_source"] = config_source
    if problem_override:
        metadata["problem_override"] = problem_override
    if hv_stop_config:
        metadata["hv_stop_config"] = hv_stop_config
    if payload.get("genealogy"):
        metadata["genealogy"] = payload["genealogy"]
    if autodiff_info is not None:
        metadata["autodiff_constraints"] = autodiff_info
    artifact_entries = {
        "fun": artifacts.get("fun"),
        "x": artifacts.get("x"),
        "g": artifacts.get("g"),
        "archive_fun": artifacts.get("archive_fun"),
        "archive_x": artifacts.get("archive_x"),
        "archive_g": artifacts.get("archive_g"),
        "genealogy": artifacts.get("genealogy"),
        "autodiff_constraints": artifacts.get("autodiff_constraints"),
        "time_ms": "time.txt",
    }
    metadata["artifacts"] = {k: v for k, v in artifact_entries.items() if v is not None}

    resolved_cfg = {
        "algorithm": algorithm_name,
        "engine": engine_name,
        "problem": selection.spec.key,
        "n_var": selection.n_var,
        "n_obj": selection.n_obj,
        "encoding": encoding,
        "population_size": config.population_size,
        "offspring_population_size": config.offspring_size(),
        "max_evaluations": config.max_evaluations,
        "seed": config.seed,
        "selection_pressure": selection_pressure,
        "external_archive_size": external_archive_size,
        "hv_threshold": hv_stop_config.get("threshold_fraction") if hv_stop_config else None,
        "hv_reference_point": hv_stop_config.get("reference_point") if hv_stop_config else None,
        "hv_reference_front": hv_stop_config.get("reference_front_path") if hv_stop_config else None,
        "nsgaii_variation": nsgaii_variation,
        "moead_variation": moead_variation,
        "smsemoa_variation": smsemoa_variation,
        "nsga3_variation": nsga3_variation,
        "config_source": config_source,
        "problem_override": problem_override,
    }
    write_metadata(output_dir, metadata, resolved_cfg)
    return artifacts, metadata, resolved_cfg


__all__ = ["print_run_banner", "build_metrics", "print_run_results", "persist_run_outputs"]
