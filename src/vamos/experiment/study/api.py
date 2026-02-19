from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from vamos.engine.tuning.ablation import AblationPlan
from vamos.experiment.services.orchestrator import run_single

from .persistence import CSVPersister
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
                nsgaii_variation=task.nsgaii_variation,
                moead_variation=task.moead_variation,
                smsemoa_variation=task.smsemoa_variation,
                config_overrides=merged,
            )
        )
    return adjusted


def run_study(
    tasks: Iterable[StudyTask],
    *,
    config_overrides: dict[str, Any] | None = None,
    mirror_output_roots: Sequence[str] | None = ("results",),
) -> list[StudyResult]:
    persister = CSVPersister(mirror_roots=mirror_output_roots) if mirror_output_roots else None
    runner = StudyRunner(persister=persister)
    overrides: dict[str, Any] = config_overrides or {}
    if overrides:
        tasks = _apply_overrides(tasks, overrides)
    return runner.run(list(tasks), run_single_fn=run_single)


def _as_dict(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return dict(value)


def build_study_tasks_from_ablation_plan(
    plan: AblationPlan,
    *,
    algorithm: str,
    base_config: Mapping[str, Any] | None = None,
    nsgaii_variations: Mapping[str, Mapping[str, Any]] | None = None,
    moead_variations: Mapping[str, Mapping[str, Any]] | None = None,
    smsemoa_variations: Mapping[str, Mapping[str, Any]] | None = None,
    engine: str | None = None,
) -> tuple[list[StudyTask], list[str]]:
    tasks: list[StudyTask] = []
    variant_names: list[str] = []
    base_cfg = base_config or {}

    for ablation_task in plan.tasks:
        overrides = ablation_task.variant.apply(base_cfg)
        overrides["max_evaluations"] = ablation_task.max_evals
        task_engine = engine or ablation_task.engine or plan.engine or "numpy"
        nsgaii_variation = _as_dict(nsgaii_variations.get(ablation_task.variant.name) if nsgaii_variations is not None else None)
        moead_variation = _as_dict(moead_variations.get(ablation_task.variant.name) if moead_variations is not None else None)
        smsemoa_variation = _as_dict(smsemoa_variations.get(ablation_task.variant.name) if smsemoa_variations is not None else None)
        tasks.append(
            StudyTask(
                algorithm=algorithm,
                engine=task_engine,
                problem=ablation_task.problem,
                seed=ablation_task.seed,
                config_overrides=overrides,
                nsgaii_variation=nsgaii_variation,
                moead_variation=moead_variation,
                smsemoa_variation=smsemoa_variation,
            )
        )
        variant_names.append(ablation_task.variant.name)

    return tasks, variant_names


def run_ablation_plan(
    plan: AblationPlan,
    *,
    algorithm: str,
    base_config: Mapping[str, Any] | None = None,
    nsgaii_variations: Mapping[str, Mapping[str, Any]] | None = None,
    moead_variations: Mapping[str, Mapping[str, Any]] | None = None,
    smsemoa_variations: Mapping[str, Mapping[str, Any]] | None = None,
    engine: str | None = None,
    config_overrides: dict[str, Any] | None = None,
    mirror_output_roots: Sequence[str] | None = ("results",),
) -> tuple[list[StudyResult], list[str]]:
    tasks, variant_names = build_study_tasks_from_ablation_plan(
        plan,
        algorithm=algorithm,
        base_config=base_config,
        nsgaii_variations=nsgaii_variations,
        moead_variations=moead_variations,
        smsemoa_variations=smsemoa_variations,
        engine=engine,
    )
    results = run_study(tasks, config_overrides=config_overrides, mirror_output_roots=mirror_output_roots)
    return results, variant_names


__all__ = ["run_study", "build_study_tasks_from_ablation_plan", "run_ablation_plan"]
