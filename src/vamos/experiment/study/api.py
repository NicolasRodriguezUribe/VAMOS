from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from vamos.experiment.services.orchestrator import run_single

from .persistence import CSVPersister
from .runner import StudyRunner, StudyResult, StudyTask


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
                external_archive_size=task.external_archive_size,
                archive_type=task.archive_type,
                nsgaii_variation=task.nsgaii_variation,
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


__all__ = ["run_study"]
