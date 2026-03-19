from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vamos.experiment.benchmark.archive_family import (
    is_archive_family_suite,
    resolve_benchmark_algorithm_alias,
    write_archive_family_summary,
)
from vamos.experiment.benchmark.suites import BenchmarkExperiment, BenchmarkSuite
from vamos.experiment.runner import run_single
from vamos.experiment.runtime.catalog import resolve_engine
from vamos.experiment.study.persistence import CSVPersister
from vamos.experiment.study.runner import StudyResult, StudyRunner, StudyTask


@dataclass
class SingleRunInfo:
    problem: str
    algorithm: str
    seed: int
    output_dir: str | None
    selection: Any
    metrics: dict[str, Any]


@dataclass
class BenchmarkResult:
    suite: BenchmarkSuite
    algorithms: list[str]
    metrics: list[str]
    base_output_dir: Path
    summary_path: Path | None
    runs: list[SingleRunInfo]
    raw_results: list[StudyResult] | None = None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _derive_budget(exp: BenchmarkExperiment, config_overrides: dict[str, Any]) -> int:
    pop = config_overrides.get("population_size")
    return exp.resolved_budget(population_size=pop)


def _prepare_tasks(
    suite: BenchmarkSuite,
    algorithms: Sequence[str],
    metrics: Sequence[str],
    base_output_dir: Path,
    global_config_overrides: dict[str, Any] | None,
) -> tuple[list[StudyTask], dict[str, Any], Path, list[str]]:
    overrides = dict(global_config_overrides or {})
    raw_root = base_output_dir / "raw_runs" / suite.name
    overrides.setdefault("output_root", str(raw_root))
    allowed_cfg_keys = {
        "title",
        "output_root",
        "population_size",
        "offspring_population_size",
        "max_evaluations",
        "seed",
        "eval_strategy",
        "n_workers",
        "live_viz",
        "live_viz_interval",
        "live_viz_max_points",
    }
    tasks: list[StudyTask] = []
    task_labels: list[str] = []
    for exp in suite.experiments:
        seeds = exp.seeds or suite.default_seeds
        for algo in algorithms:
            alias = resolve_benchmark_algorithm_alias(algo)
            execution_algorithm = alias.execution_algorithm if alias is not None else algo
            for seed in seeds:
                cfg = {k: v for k, v in overrides.items() if k in allowed_cfg_keys}
                cfg["max_evaluations"] = _derive_budget(exp, cfg)
                if alias is not None:
                    cfg["output_root"] = str(raw_root / alias.output_root_suffix)
                tasks.append(
                    StudyTask(
                        algorithm=execution_algorithm,
                        engine=resolve_engine(overrides.get("engine"), algorithm=execution_algorithm),
                        problem=exp.problem_name,
                        n_var=exp.problem_params.get("n_var"),
                        n_obj=exp.problem_params.get("n_obj"),
                        seed=seed,
                        config_overrides=cfg,
                        nsgaii_variation=dict(alias.nsgaii_variation) if alias is not None and alias.nsgaii_variation is not None else None,
                    )
                )
                task_labels.append(alias.label if alias is not None else execution_algorithm)
    return tasks, overrides, raw_root, task_labels


def run_benchmark_suite(
    suite: BenchmarkSuite,
    algorithms: Sequence[str] | None,
    metrics: Sequence[str] | None,
    base_output_dir: Path,
    global_config_overrides: dict[str, Any] | None = None,
    *,
    study_runner_cls: type[StudyRunner] = StudyRunner,
) -> BenchmarkResult:
    algos = list(algorithms) if algorithms else list(suite.default_algorithms)
    metric_list = list(metrics) if metrics else list(suite.default_metrics)
    base_output_dir = base_output_dir.resolve()
    tasks, overrides, raw_root, task_labels = _prepare_tasks(suite, algos, metric_list, base_output_dir, global_config_overrides)
    # HV is computed separately. Archive-subset indicator names map back to the
    # same base indicator computation and are exported as separate columns.
    indicator_metrics: list[str] = []
    for metric in metric_list:
        name = metric.lower()
        if name in {"hv", "hypervolume", "archive_subset_hv"}:
            continue
        if name.startswith("archive_subset_"):
            name = name.removeprefix("archive_subset_")
        if name not in indicator_metrics:
            indicator_metrics.append(name)
    persister = CSVPersister(mirror_roots=())
    runner = study_runner_cls(verbose=True, indicators=indicator_metrics, persister=persister)
    summary_dir = _ensure_dir(base_output_dir / "summary")
    results = runner.run(
        tasks,
        run_single_fn=run_single,
    )
    for res, label in zip(results, task_labels):
        res.metrics["algorithm_base"] = res.task.algorithm
        res.metrics["algorithm"] = label
    persister.save_results(results, summary_dir / "metrics.csv")
    archive_family_requested = is_archive_family_suite(suite.name) or any(resolve_benchmark_algorithm_alias(name) is not None for name in algos)
    archive_family_artifacts = write_archive_family_summary(results, summary_dir) if archive_family_requested else {}
    runs: list[SingleRunInfo] = []
    for res in results:
        algorithm_name = res.metrics.get("algorithm") or res.task.algorithm
        runs.append(
            SingleRunInfo(
                problem=res.selection.spec.key,
                algorithm=str(algorithm_name),
                seed=res.task.seed,
                output_dir=res.metrics.get("output_dir"),
                selection=res.selection,
                metrics=res.metrics,
            )
        )

    meta = {
        "suite": suite.name,
        "description": suite.description,
        "algorithms": algos,
        "metrics": metric_list,
        "config_overrides": overrides,
        "experiments": [
            {
                "problem": exp.problem_name,
                "params": exp.problem_params,
                "evaluation_budget": exp.evaluation_budget,
                "max_generations": exp.max_generations,
                "seeds": exp.seeds or suite.default_seeds,
            }
            for exp in suite.experiments
        ],
    }
    if archive_family_artifacts:
        meta["archive_family_artifacts"] = {name: str(path) for name, path in archive_family_artifacts.items()}
    meta_path = summary_dir / "suite.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return BenchmarkResult(
        suite=suite,
        algorithms=algos,
        metrics=metric_list,
        base_output_dir=base_output_dir,
        summary_path=summary_dir / "metrics.csv",
        runs=runs,
        raw_results=results,
    )
