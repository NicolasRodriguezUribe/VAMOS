from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Iterable, List, Sequence, Any, Dict

import numpy as np

from vamos.engine.algorithm.components.hypervolume import hypervolume
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.core.hv_stop import compute_hv_reference
from vamos.foundation.problem.registry import ProblemSelection, make_problem_selection
from vamos.foundation.core.runner import run_single
from vamos.foundation.metrics.moocore_indicators import has_moocore, get_indicator, HVIndicator
from vamos.foundation.kernel.numpy_backend import _fast_non_dominated_sort


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
    seed: int = ExperimentConfig().seed
    selection_pressure: int = 2
    config_overrides: Dict[str, Any] | None = None


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
        row = {
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
        for name, value in (self.metrics.get("indicator_values") or {}).items():
            row[f"indicator_{name}"] = value
        return row


class StudyRunner:
    """
    Executes batches of study tasks and emits structured summaries.
    """

    def __init__(
        self,
        *,
        verbose: bool = True,
        mirror_output_roots: Sequence[str | Path] | None = ("results",),
        indicators: Sequence[str] | None = ("hv",),
    ):
        self.verbose = verbose
        roots = mirror_output_roots or ()
        self._mirror_roots = tuple(Path(root) for root in roots)
        self.indicators = tuple(indicators or ())

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
            cfg_kwargs = {"seed": task.seed}
            if task.config_overrides:
                cfg_kwargs.update(task.config_overrides)
            task_config = ExperimentConfig(**cfg_kwargs)
            metrics = run_single(
                task.engine,
                task.algorithm,
                selection,
                task_config,
                selection_pressure=task.selection_pressure,
            )
            self._mirror_artifacts(metrics)
            results.append(StudyResult(task=task, selection=selection, metrics=metrics))

        self._attach_hypervolume(results)
        self._attach_indicators(results)
        if export_csv_path is not None:
            self.export_csv(results, export_csv_path)
        return results

    def _attach_hypervolume(self, results: Iterable[StudyResult]) -> None:
        fronts = [res.metrics["F"] for res in results]
        hv_ref_point = compute_hv_reference(fronts)
        for res in results:
            metrics = res.metrics
            backend = metrics.pop("_kernel_backend", None)
            if backend and backend.supports_quality_indicator("hypervolume"):
                hv_val = backend.hypervolume(metrics["F"], hv_ref_point)
                hv_source = backend.__class__.__name__
            else:
                hv_val = hypervolume(metrics["F"], hv_ref_point)
                hv_source = "global"
            metrics["hv"] = hv_val
            metrics["hv_source"] = hv_source
            metrics["hv_reference"] = hv_ref_point

    @staticmethod
    def _nondominated_union(fronts: list[np.ndarray]) -> np.ndarray:
        if not fronts:
            return np.empty((0, 0))
        F = np.vstack(fronts)
        if F.size == 0:
            return F
        fronts_idx, _ = _fast_non_dominated_sort(F)
        if not fronts_idx:
            return F
        first = fronts_idx[0]
        return F[first] if first else F

    def _attach_indicators(self, results: Iterable[StudyResult]) -> None:
        if not self.indicators:
            return
        if not has_moocore():
            if self.verbose:
                print("[Study] MooCore not available; skipping indicator computation.")
            return
        fronts = [res.metrics["F"] for res in results]
        ref_front = self._nondominated_union(fronts)
        hv_ref_point = compute_hv_reference(fronts)
        for res in results:
            vals = {}
            for name in self.indicators:
                try:
                    if name in {"hv", "hypervolume"}:
                        indicator = HVIndicator(reference_point=hv_ref_point)
                        vals[name] = indicator.compute(res.metrics["F"]).value
                    elif name in {"igd", "igd+", "igd_plus", "epsilon_additive", "epsilon_mult", "avg_hausdorff"}:
                        indicator = get_indicator(name, reference_front=ref_front)
                        vals[name] = indicator.compute(res.metrics["F"]).value
                    else:
                        indicator = get_indicator(name)
                        vals[name] = indicator.compute(res.metrics["F"]).value
                except Exception as exc:
                    if self.verbose:
                        print(f"[Study] indicator '{name}' failed: {exc}")
                    vals[name] = None
            res.metrics["indicator_values"] = vals

    def export_csv(self, results: Iterable[StudyResult], path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [res.to_row() for res in results]
        if not rows:
            return path
        fieldnames: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        if self.verbose:
            print(f"[Study] CSV exported to {path}")
        return path

    def _mirror_artifacts(self, metrics: dict) -> None:
        if not self._mirror_roots:
            return
        output_dir = metrics.get("output_dir")
        if not output_dir:
            return
        src_dir = Path(output_dir).resolve()
        base_root = Path(ExperimentConfig().output_root).resolve()
        relative = None
        try:
            relative = src_dir.relative_to(base_root)
        except ValueError:
            relative = None

        for root in self._mirror_roots:
            target_root = Path(root).resolve()
            if relative is not None:
                dst = target_root / relative
            else:
                dst = target_root / src_dir.name
            if dst.resolve() == src_dir:
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("FUN.csv", "ARCHIVE_FUN.csv", "time.txt", "metadata.json"):
                src_file = src_dir / name
                if src_file.exists():
                    copy2(src_file, dst / name)

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
                sel_pressure = base.selection_pressure
            else:
                problem = problem_entry["problem"]
                n_var = problem_entry.get("n_var")
                n_obj = problem_entry.get("n_obj")
                sel_pressure = problem_entry.get("selection_pressure", 2)
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
                                selection_pressure=sel_pressure,
                            )
                        )
        return entries
