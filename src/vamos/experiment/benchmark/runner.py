"""Benchmark execution through canonical durable studies."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vamos.experiment.benchmark.suites import BenchmarkExperiment, BenchmarkSuite
from vamos.experiment.study_analysis import SummarySource, derive_summary_rows
from vamos.study_artifacts import Study, StudyReport, StudySpec, StudySummary, create_study, load_study, plan_study


@dataclass(frozen=True, slots=True)
class BenchmarkStudy:
    """One homogeneous canonical study belonging to a benchmark suite."""

    experiment_index: int
    study: Study
    report: StudyReport
    summary: StudySummary


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    suite: BenchmarkSuite
    algorithms: tuple[str, ...]
    metrics: tuple[str, ...]
    base_output_dir: Path
    summary_path: Path | None
    studies: tuple[BenchmarkStudy, ...]

    @property
    def study_ids(self) -> tuple[str, ...]:
        return tuple(item.study.study_id for item in self.studies)

    @property
    def study_roots(self) -> tuple[Path, ...]:
        return tuple(item.study.root for item in self.studies)

    def summary_rows(self) -> tuple[dict[str, Any], ...]:
        """Render derived benchmark rows from canonical summaries."""
        sources = tuple(SummarySource(item.study.root, item.summary, {}) for item in self.studies)
        return derive_summary_rows(sources, indicators=self.metrics)


def run_benchmark_suite(
    suite: BenchmarkSuite,
    algorithms: Sequence[str] | None,
    metrics: Sequence[str] | None,
    base_output_dir: Path,
    global_config_overrides: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Run each benchmark experiment as one homogeneous canonical study."""
    algos = tuple(algorithms) if algorithms else tuple(suite.default_algorithms)
    metric_names = tuple(metrics) if metrics else tuple(suite.default_metrics)
    root = base_output_dir.absolute()
    overrides = dict(global_config_overrides or {})
    executions = tuple(
        _run_experiment_study(suite, experiment, index, algos, root, overrides) for index, experiment in enumerate(suite.experiments)
    )
    provisional = BenchmarkResult(suite, algos, metric_names, root, None, executions)
    summary_dir = root / "summary"
    summary_path = summary_dir / "metrics.csv"
    _write_summary(provisional.summary_rows(), summary_path)
    _write_suite_metadata(provisional, summary_dir / "suite.json", overrides)
    return BenchmarkResult(suite, algos, metric_names, root, summary_path, executions)


def load_benchmark_result(
    suite: BenchmarkSuite,
    *,
    algorithms: Sequence[str],
    metrics: Sequence[str],
    base_output_dir: Path,
) -> BenchmarkResult:
    """Regenerate benchmark views by loading only canonical study roots."""
    root = base_output_dir.absolute()
    studies: list[BenchmarkStudy] = []
    for manifest_path in root.rglob("study-manifest.json"):
        study = load_study(manifest_path.parent)
        labels = study.spec.labels or {}
        if labels.get("workflow") != "benchmark" or labels.get("suite") != suite.name:
            continue
        index = (study.spec.metadata or {}).get("experiment_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Benchmark study metadata requires an integer experiment_index.")
        studies.append(BenchmarkStudy(index, study, study.inspect(), study.summarize()))
    studies.sort(key=lambda item: item.experiment_index)
    return BenchmarkResult(
        suite,
        tuple(algorithms),
        tuple(metrics),
        root,
        root / "summary" / "metrics.csv",
        tuple(studies),
    )


def _run_experiment_study(
    suite: BenchmarkSuite,
    experiment: BenchmarkExperiment,
    index: int,
    algorithms: tuple[str, ...],
    root: Path,
    overrides: Mapping[str, Any],
) -> BenchmarkStudy:
    pop_size = _optional_int(overrides.get("population_size"), field="population_size")
    budget = experiment.resolved_budget(population_size=pop_size)
    problem_kwargs = {key: value for key, value in experiment.problem_params.items() if key not in {"n_var", "n_obj"}}
    spec = StudySpec(
        problems=[experiment.problem_name],
        algorithms=algorithms,
        seeds=experiment.seeds or suite.default_seeds,
        max_evaluations=budget,
        pop_size=pop_size,
        engine=_optional_string(overrides.get("engine")),
        eval_strategy=str(overrides.get("eval_strategy", "serial")),
        n_var=_optional_int(experiment.problem_params.get("n_var"), field="n_var"),
        n_obj=_optional_int(experiment.problem_params.get("n_obj"), field="n_obj"),
        problem_kwargs=problem_kwargs,
        algorithm_configs=_algorithm_configs(algorithms, overrides),
        on_error="continue",
        labels={"workflow": "benchmark", "suite": suite.name},
        metadata={"experiment_index": index},
    )
    destination = root / f"study-{index:04d}"
    planned = plan_study(spec, output=destination)
    created = create_study(spec, output=destination)
    if (planned.plan_id, planned.task_ids) != (created.plan_id, tuple(item.task_id for item in created.plan.tasks)):
        raise AssertionError("benchmark planning and creation identities differ")
    completed = created.run()
    return BenchmarkStudy(index, completed, completed.inspect(), completed.summarize())


def _algorithm_configs(algorithms: tuple[str, ...], overrides: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    offspring = _optional_int(overrides.get("offspring_population_size"), field="offspring_population_size")
    if offspring is None:
        return {}
    return {algorithm: {"offspring_size": offspring} for algorithm in algorithms if algorithm.strip().lower() not in {"moead", "smpso"}}


def _write_summary(rows: tuple[dict[str, Any], ...], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_suite_metadata(result: BenchmarkResult, path: Path, overrides: Mapping[str, Any]) -> None:
    payload = {
        "suite": result.suite.name,
        "description": result.suite.description,
        "algorithms": list(result.algorithms),
        "metrics": list(result.metrics),
        "study_ids": list(result.study_ids),
        "plan_ids": [item.study.plan_id for item in result.studies],
        "config_overrides": {key: value for key, value in overrides.items() if key != "output_root"},
        "experiments": [
            {
                "problem": experiment.problem_name,
                "params": experiment.problem_params,
                "evaluation_budget": experiment.evaluation_budget,
                "max_generations": experiment.max_generations,
                "seeds": list(experiment.seeds or result.suite.default_seeds),
            }
            for experiment in result.suite.experiments
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer when provided.")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("engine must be a non-empty string when provided.")
    return value


__all__ = ["BenchmarkResult", "BenchmarkStudy", "load_benchmark_result", "run_benchmark_suite"]
