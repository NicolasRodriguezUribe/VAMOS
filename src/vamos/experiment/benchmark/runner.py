from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from vamos.experiment.benchmark.suites import BenchmarkSuite, BenchmarkExperiment
from vamos.foundation.core.experiment_config import DEFAULT_ENGINE
from vamos.experiment.study.runner import StudyRunner, StudyTask, StudyResult
from vamos.experiment.runner import run_single


@dataclass
class SingleRunInfo:
    problem: str
    algorithm: str
    seed: int
    output_dir: str | None
    selection: Any
    metrics: dict


@dataclass
class BenchmarkResult:
    suite: BenchmarkSuite
    algorithms: List[str]
    metrics: List[str]
    base_output_dir: Path
    summary_path: Path | None
    runs: List[SingleRunInfo]
    raw_results: List[StudyResult] | None = None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _derive_budget(exp: BenchmarkExperiment, config_overrides: Dict[str, Any]) -> int:
    pop = config_overrides.get("population_size")
    return exp.resolved_budget(population_size=pop)


def _prepare_tasks(
    suite: BenchmarkSuite,
    algorithms: Sequence[str],
    metrics: Sequence[str],
    base_output_dir: Path,
    global_config_overrides: Dict[str, Any] | None,
) -> tuple[list[StudyTask], Dict[str, Any], Path]:
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
        "eval_backend",
        "n_workers",
        "live_viz",
        "live_viz_interval",
        "live_viz_max_points",
    }
    tasks: list[StudyTask] = []
    for exp in suite.experiments:
        seeds = exp.seeds or suite.default_seeds
        for algo in algorithms:
            for seed in seeds:
                cfg = {k: v for k, v in overrides.items() if k in allowed_cfg_keys}
                cfg["max_evaluations"] = _derive_budget(exp, cfg)
                tasks.append(
                    StudyTask(
                        algorithm=algo,
                        engine=overrides.get("engine", DEFAULT_ENGINE),
                        problem=exp.problem_name,
                        n_var=exp.problem_params.get("n_var"),
                        n_obj=exp.problem_params.get("n_obj"),
                        seed=seed,
                        config_overrides=cfg,
                    )
                )
    return tasks, overrides, raw_root


def run_benchmark_suite(
    suite: BenchmarkSuite,
    algorithms: Sequence[str] | None,
    metrics: Sequence[str] | None,
    base_output_dir: Path,
    global_config_overrides: Dict[str, Any] | None = None,
    *,
    study_runner_cls: type[StudyRunner] = StudyRunner,
) -> BenchmarkResult:
    algos = list(algorithms) if algorithms else list(suite.default_algorithms)
    metric_list = list(metrics) if metrics else list(suite.default_metrics)
    base_output_dir = base_output_dir.resolve()
    tasks, overrides, raw_root = _prepare_tasks(suite, algos, metric_list, base_output_dir, global_config_overrides)
    # hv computed separately; ask for additional indicators only
    indicator_metrics = [m for m in metric_list if m.lower() not in {"hv", "hypervolume"}]
    runner = study_runner_cls(verbose=True, mirror_output_roots=(), indicators=indicator_metrics)
    summary_dir = _ensure_dir(base_output_dir / "summary")
    results = runner.run(
        tasks,
        export_csv_path=summary_dir / "metrics.csv",
        run_single_fn=run_single,
    )
    runs: List[SingleRunInfo] = []
    for res in results:
        runs.append(
            SingleRunInfo(
                problem=res.selection.spec.key,
                algorithm=res.metrics.get("algorithm"),
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
