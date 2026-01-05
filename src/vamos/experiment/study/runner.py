from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Any, Dict, Protocol

import numpy as np

from vamos.foundation.metrics.hypervolume import hypervolume
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.core.hv_stop import compute_hv_reference
from vamos.foundation.problem.registry import ProblemSelection, make_problem_selection
from vamos.foundation.metrics.moocore_indicators import has_moocore, get_indicator, HVIndicator
from vamos.foundation.kernel.numpy_backend import _fast_non_dominated_sort

from vamos.experiment.study.types import StudyTask, StudyResult
from vamos.experiment.study.persistence import StudyPersister


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class StudyRunner:
    """
    Executes batches of study tasks and emits structured summaries.
    """

    def __init__(
        self,
        *,
        verbose: bool = True,
        indicators: Sequence[str] | None = ("hv",),
        persister: StudyPersister | None = None,
    ):
        self.verbose = verbose
        self.indicators = tuple(indicators or ())
        self.persister = persister

    def run(
        self,
        tasks: Sequence[StudyTask],
        *,
        export_csv_path: str | Path | None = None, # Deprecated but kept for compat with wiring
        run_single_fn: Callable[..., dict] | None = None,
    ) -> List[StudyResult]:
        if not tasks:
            return []
        if run_single_fn is None:
            raise ValueError("run_single_fn is required to execute StudyRunner tasks.")
        
        results: List[StudyResult] = []
        for idx, task in enumerate(tasks, start=1):
            if self.verbose:
                _logger().info(
                    "[Study] (%s/%s) %s | %s | %s | seed=%s",
                    idx,
                    len(tasks),
                    task.algorithm,
                    task.engine,
                    task.problem,
                    task.seed,
                )
            selection = make_problem_selection(task.problem, n_var=task.n_var, n_obj=task.n_obj)
            cfg_kwargs = {"seed": task.seed}
            if task.config_overrides:
                cfg_kwargs.update(task.config_overrides)
            task_config = ExperimentConfig(**cfg_kwargs)
            metrics = run_single_fn(
                task.engine,
                task.algorithm,
                selection,
                task_config,
                external_archive_size=task.external_archive_size,
                archive_type=task.archive_type,
                selection_pressure=task.selection_pressure,
                nsgaii_variation=task.nsgaii_variation,
            )
            
            result = StudyResult(task=task, selection=selection, metrics=metrics)
            results.append(result)
            
            # Delegate artifact mirroring to persister if available
            if self.persister:
                self.persister.mirror_artifacts(result)
            

        self._attach_hypervolume(results)
        self._attach_indicators(results)
        
        # Delegate CSV saving
        if self.persister and export_csv_path:
             self.persister.save_results(results, export_csv_path)
             
        return results

    def _attach_hypervolume(self, results: Iterable[StudyResult]) -> None:
        fronts = [res.metrics["F"] for res in results]
        if not fronts:
            return
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
                _logger().info("[Study] MooCore not available; skipping indicator computation.")
            return
        fronts = [res.metrics["F"] for res in results]
        if not fronts:
             return
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
                        _logger().warning("[Study] indicator '%s' failed: %s", name, exc)
                    vals[name] = None
            res.metrics["indicator_values"] = vals
            
    # expand_tasks is static and purely data transformation, can stay or move.
    # Moving it to types or keeping it here as helper is fine. Keeping for backward compat.
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
            # Import locally to avoid circular dep if types wasn't imported (though it is)
            # Actually StudyTask is imported at top.
            if isinstance(problem_entry, StudyTask):
                base = problem_entry
                problem = base.problem
                n_var = base.n_var
                n_obj = base.n_obj
                sel_pressure = base.selection_pressure
                external_archive_size = base.external_archive_size
                archive_type = base.archive_type
                nsgaii_variation = base.nsgaii_variation
            else:
                problem = problem_entry["problem"]
                n_var = problem_entry.get("n_var")
                n_obj = problem_entry.get("n_obj")
                sel_pressure = problem_entry.get("selection_pressure", 2)
                external_archive_size = problem_entry.get("external_archive_size")
                archive_type = problem_entry.get("archive_type", "hypervolume")
                nsgaii_variation = problem_entry.get("nsgaii_variation")
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
                                external_archive_size=external_archive_size,
                                archive_type=archive_type,
                                nsgaii_variation=nsgaii_variation,
                            )
                        )
        return entries
