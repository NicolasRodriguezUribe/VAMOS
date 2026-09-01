"""Canonical durable-study execution for ablation plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vamos.engine.tuning.ablation import AblationPlan, AblationTask
from vamos.experiment._execution_support import VariationConfigs
from vamos.experiment.study_analysis import SummarySource, derive_summary_rows
from vamos.study_artifacts import Study, StudyReport, StudySpec, StudySummary, create_study, plan_study


@dataclass(frozen=True, slots=True)
class AblationStudy:
    """One homogeneous canonical study within an ablation execution."""

    variant: str
    problem: str
    study: Study
    report: StudyReport
    summary: StudySummary


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Ablation-specific grouping of canonical study projections."""

    studies: tuple[AblationStudy, ...]

    @property
    def study_ids(self) -> tuple[str, ...]:
        return tuple(item.study.study_id for item in self.studies)

    @property
    def study_roots(self) -> tuple[Path, ...]:
        return tuple(item.study.root for item in self.studies)

    def summary_rows(self) -> tuple[dict[str, Any], ...]:
        """Return derived rows while retaining canonical task/run evidence."""
        sources = tuple(SummarySource(item.study.root, item.summary, {"variant": item.variant}) for item in self.studies)
        return derive_summary_rows(sources, indicators=("hv",))


def run_ablation_plan(
    plan: AblationPlan,
    *,
    algorithm: str,
    output: str | Path,
    base_config: Mapping[str, Any] | None = None,
    variations_by_variant: Mapping[str, VariationConfigs] | None = None,
    engine: str | None = None,
) -> AblationResult:
    """Execute each scientifically homogeneous ablation group as a study."""
    groups = _homogeneous_groups(plan)
    root = Path(output).absolute()
    executions: list[AblationStudy] = []
    for index, tasks in enumerate(groups):
        first = tasks[0]
        spec = _study_spec(
            plan,
            tasks,
            algorithm=algorithm,
            base_config=base_config,
            variations_by_variant=variations_by_variant,
            engine=engine,
        )
        destination = root / f"study-{index:04d}"
        planned = plan_study(spec, output=destination)
        created = create_study(spec, output=destination)
        if (planned.plan_id, planned.task_ids) != (created.plan_id, tuple(item.task_id for item in created.plan.tasks)):
            raise AssertionError("ablation planning and creation identities differ")
        completed = created.run()
        executions.append(
            AblationStudy(
                variant=first.variant.name,
                problem=first.problem,
                study=completed,
                report=completed.inspect(),
                summary=completed.summarize(),
            )
        )
    return AblationResult(tuple(executions))


def _homogeneous_groups(plan: AblationPlan) -> tuple[tuple[AblationTask, ...], ...]:
    groups: list[list[AblationTask]] = []
    keys: list[tuple[str, str, int, str | None]] = []
    for task in plan.tasks:
        key = (task.problem, task.variant.name, task.max_evals, task.engine or plan.engine)
        if key not in keys:
            keys.append(key)
            groups.append([])
        groups[keys.index(key)].append(task)
    return tuple(tuple(group) for group in groups)


def _study_spec(
    plan: AblationPlan,
    tasks: tuple[AblationTask, ...],
    *,
    algorithm: str,
    base_config: Mapping[str, Any] | None,
    variations_by_variant: Mapping[str, VariationConfigs] | None,
    engine: str | None,
) -> StudySpec:
    first = tasks[0]
    config = first.variant.apply(base_config)
    variation = variations_by_variant.get(first.variant.name) if variations_by_variant is not None else None
    algorithm_config = _algorithm_config(config, variation, algorithm)
    return StudySpec(
        problems=[first.problem],
        algorithms=[algorithm],
        seeds=[task.seed for task in tasks],
        max_evaluations=first.max_evals,
        pop_size=_optional_int(config.get("population_size"), field="population_size"),
        engine=engine or first.engine or plan.engine,
        eval_strategy=str(config.get("eval_strategy", "serial")),
        algorithm_configs={algorithm: algorithm_config},
        on_error="fail_fast",
        labels={"workflow": "ablation", "variant": first.variant.name},
        metadata={
            "ablation": plan.metadata,
            "variant": {
                "name": first.variant.name,
                "label": first.variant.label or first.variant.name,
                "tags": list(first.variant.tags),
            },
        },
    )


def _algorithm_config(config: Mapping[str, Any], variation: VariationConfigs | None, algorithm: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    offspring = config.get("offspring_population_size")
    if offspring is not None and algorithm.strip().lower() not in {"moead", "smpso"}:
        result["offspring_size"] = _optional_int(offspring, field="offspring_population_size")
    if variation is not None:
        selected = variation.copy_field(algorithm.strip().lower())
        if selected is not None:
            result.update(selected)
    return result


def _optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer when provided.")
    return value


__all__ = ["AblationResult", "AblationStudy", "run_ablation_plan"]
