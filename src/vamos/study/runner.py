from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from vamos import main as vamos_main
from vamos.problem.registry import ProblemSelection, make_problem_selection


@dataclass(frozen=True)
class StudyTask:
    """
    Defines a single algorithm/engine/problem/seed combination.
    """

    algorithm: str
    engine: str
    problem: str
    n_var: int | None = None
    n_obj: int | None = None
    seed: int = vamos_main.SEED


@dataclass
class StudyResult:
    task: StudyTask
    selection: ProblemSelection
    metrics: dict

    def to_row(self) -> dict:
        hv_ref = self.metrics.get("hv_reference")
        hv_ref_str = (
            " ".join(f"{val:.6f}" for val in hv_ref) if isinstance(hv_ref, np.ndarray) else ""
        )
        return {
            "problem": self.selection.spec.key,
            "problem_label": self.selection.spec.label,
            "n_var": self.selection.n_var,
            "n_obj": self.selection.n_obj,
            "algorithm": self.metrics["algorithm"],
            "engine": self.metrics["engine"],
            "seed": self.task.seed,
            "time_ms": self.metrics["time_ms"],
            "evaluations": self.metrics["evaluations"],
            "evals_per_sec": self.metrics["evals_per_sec"],
            "spread": self.metrics.get("spread"),
            "hv": self.metrics.get("hv"),
            "hv_source": self.metrics.get("hv_source"),
            "hv_reference": hv_ref_str,
            "backend_device": self.metrics.get("backend_device"),
            "backend_capabilities": ",".join(self.metrics.get("backend_capabilities", [])),
            "output_dir": self.metrics.get("output_dir"),
        }


class StudyRunner:
    """
    Executes batches of study tasks and emits structured summaries.
    """

    def __init__(self, *, verbose: bool = True):
        self.verbose = verbose

    def run(
        self,
        tasks: Sequence[StudyTask],
        *,
        export_csv_path: str | Path | None = None,
    ) -> List[StudyResult]:
        if not tasks:
            return []
        results: List[StudyResult] = []
        for idx, task in enumerate(tasks, start=1):
            if self.verbose:
                print(
                    f"[Study] ({idx}/{len(tasks)}) "
                    f"{task.algorithm} | {task.engine} | {task.problem} | seed={task.seed}"
                )
            selection = make_problem_selection(
                task.problem, n_var=task.n_var, n_obj=task.n_obj
            )
            metrics = vamos_main._run_single(
                task.engine, task.algorithm, selection, seed=task.seed
            )
            results.append(StudyResult(task=task, selection=selection, metrics=metrics))

        self._attach_hypervolume(results)
        if export_csv_path is not None:
            self.export_csv(results, export_csv_path)
        return results

    def _attach_hypervolume(self, results: Iterable[StudyResult]) -> None:
        fronts = [res.metrics["F"] for res in results]
        hv_ref_point = vamos_main._compute_hv_reference(fronts)
        for res in results:
            metrics = res.metrics
            backend = metrics.pop("_kernel_backend", None)
            if backend and backend.supports_quality_indicator("hypervolume"):
                hv_val = backend.hypervolume(metrics["F"], hv_ref_point)
                hv_source = backend.__class__.__name__
            else:
                hv_val = vamos_main.hypervolume(metrics["F"], hv_ref_point)
                hv_source = "global"
            metrics["hv"] = hv_val
            metrics["hv_source"] = hv_source
            metrics["hv_reference"] = hv_ref_point

    def export_csv(self, results: Iterable[StudyResult], path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [res.to_row() for res in results]
        if not rows:
            return path
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        if self.verbose:
            print(f"[Study] CSV exported to {path}")
        return path

    @staticmethod
    def expand_tasks(
        problems: Sequence[StudyTask | dict],
        algorithms: Sequence[str],
        engines: Sequence[str],
        seeds: Sequence[int],
    ) -> List[StudyTask]:
        """
        Convenience helper: expand grid definitions into StudyTask objects.
        Each entry in `problems` can be either a StudyTask (with engine/algorithm ignored)
        or a dict containing keys {problem, n_var, n_obj}.
        """
        entries: List[StudyTask] = []
        for problem_entry in problems:
            if isinstance(problem_entry, StudyTask):
                base = problem_entry
                problem = base.problem
                n_var = base.n_var
                n_obj = base.n_obj
            else:
                problem = problem_entry["problem"]
                n_var = problem_entry.get("n_var")
                n_obj = problem_entry.get("n_obj")
            for algorithm in algorithms:
                for engine in engines:
                    for seed in seeds:
                        entries.append(
                            StudyTask(
                                algorithm=algorithm,
                                engine=engine,
                                problem=problem,
                                n_var=n_var,
                                n_obj=n_obj,
                                seed=seed,
                            )
                        )
        return entries
