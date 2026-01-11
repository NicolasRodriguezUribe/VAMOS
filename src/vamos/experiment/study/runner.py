from __future__ import annotations

import logging
import inspect
from typing import Callable, Iterable, List, Sequence, Any

import numpy as np

from vamos.foundation.metrics.hypervolume import hypervolume
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.core.hv_stop import compute_hv_reference
from vamos.foundation.problem.registry import make_problem_selection
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
        evaluator: Any | None = None,
        termination: Any | None = None,
    ):
        self.verbose = verbose
        self.indicators = tuple(indicators or ())
        self.persister = persister
        self.evaluator = evaluator
        self.termination = termination

    def run(
        self,
        tasks: Sequence[StudyTask],
        *,
        run_single_fn: Callable[..., dict] | None = None,
    ) -> List[StudyResult]:
        if not tasks:
            return []
        if run_single_fn is None:
            raise ValueError("run_single_fn is required to execute StudyRunner tasks.")

        results: List[StudyResult] = []
        run_sig = inspect.signature(run_single_fn)
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
            extra_kwargs: dict[str, Any] = {}
            if self.evaluator is not None and "evaluator" in run_sig.parameters:
                extra_kwargs["evaluator"] = self.evaluator
            if self.termination is not None and "termination" in run_sig.parameters:
                extra_kwargs["termination"] = self.termination
            metrics = run_single_fn(
                task.engine,
                task.algorithm,
                selection,
                task_config,
                external_archive_size=task.external_archive_size,
                archive_type=task.archive_type,
                selection_pressure=task.selection_pressure,
                nsgaii_variation=task.nsgaii_variation,
                **extra_kwargs,
            )

            result = StudyResult(task=task, selection=selection, metrics=metrics)
            results.append(result)

            # Delegate artifact mirroring to persister if available
            if self.persister:
                self.persister.mirror_artifacts(result)

        self._attach_hypervolume(results)
        self._attach_indicators(results)

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


__all__ = ["StudyRunner", "StudyTask", "StudyResult"]
