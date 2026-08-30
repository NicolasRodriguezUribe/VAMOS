from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from vamos.engine.tuning.ablation import AblationPlan
from vamos.experiment._execution_support import VariationConfigs
from vamos.experiment.services.orchestrator import run_single

from .runner import StudyResult, StudyRunner, StudyTask


def _apply_overrides(tasks: Iterable[StudyTask], overrides: dict[str, Any]) -> list[StudyTask]:
    adjusted: list[StudyTask] = []
    for task in tasks:
        merged: dict[str, Any] = dict(task.config_overrides or {})
        merged.update({k: v for k, v in overrides.items() if v is not None})
        adjusted.append(
            StudyTask(
                algorithm=task.algorithm,
                engine=task.engine,
                problem=task.problem,
                n_var=task.n_var,
                n_obj=task.n_obj,
                seed=task.seed,
                selection_pressure=task.selection_pressure,
                external_archive=task.external_archive,
                variations=task.variations.copy() if task.variations is not None else None,
                config_overrides=merged,
            )
        )
    return adjusted


def run_study(
    tasks: Iterable[StudyTask],
    *,
    config_overrides: dict[str, Any] | None = None,
) -> list[StudyResult]:
    runner = StudyRunner()
    overrides: dict[str, Any] = config_overrides or {}
    if overrides:
        tasks = _apply_overrides(tasks, overrides)
    return runner.run(list(tasks), run_single_fn=run_single)


def build_study_tasks_from_ablation_plan(
    plan: AblationPlan,
    *,
    algorithm: str,
    base_config: Mapping[str, Any] | None = None,
    variations_by_variant: Mapping[str, VariationConfigs] | None = None,
    engine: str | None = None,
) -> tuple[list[StudyTask], list[str]]:
    tasks: list[StudyTask] = []
    variant_names: list[str] = []
    base_cfg = base_config or {}

    for ablation_task in plan.tasks:
        overrides = ablation_task.variant.apply(base_cfg)
        overrides["max_evaluations"] = ablation_task.max_evals
        task_engine = engine or ablation_task.engine or plan.engine or "numpy"
        task_variations = None
        if variations_by_variant is not None:
            variant_variations = variations_by_variant.get(ablation_task.variant.name)
            if variant_variations is not None:
                task_variations = variant_variations.copy()
        tasks.append(
            StudyTask(
                algorithm=algorithm,
                engine=task_engine,
                problem=ablation_task.problem,
                seed=ablation_task.seed,
                config_overrides=overrides,
                variations=task_variations,
            )
        )
        variant_names.append(ablation_task.variant.name)

    return tasks, variant_names


def run_ablation_plan(
    plan: AblationPlan,
    *,
    algorithm: str,
    base_config: Mapping[str, Any] | None = None,
    variations_by_variant: Mapping[str, VariationConfigs] | None = None,
    engine: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> tuple[list[StudyResult], list[str]]:
    tasks, variant_names = build_study_tasks_from_ablation_plan(
        plan,
        algorithm=algorithm,
        base_config=base_config,
        variations_by_variant=variations_by_variant,
        engine=engine,
    )
    results = run_study(tasks, config_overrides=config_overrides)
    return results, variant_names


__all__ = ["run_study", "build_study_tasks_from_ablation_plan", "run_ablation_plan"]
