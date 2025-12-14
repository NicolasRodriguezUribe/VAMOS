"""
Central orchestration helpers for experiments.

These thin wrappers keep CLI, benchmark tools, and zoo presets aligned around
the same execution path.
"""
from __future__ import annotations

from argparse import Namespace
from typing import Iterable, Sequence

from vamos.experiment.study.runner import StudyRunner, StudyTask, StudyResult
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.core.runner import run_from_args, run_single
from vamos.foundation.problem.registry import make_problem_selection


def run_experiment(
    *,
    algorithm: str,
    problem: str,
    engine: str = "numpy",
    config: ExperimentConfig | None = None,
    n_var: int | None = None,
    n_obj: int | None = None,
    selection_pressure: int = 2,
    **kwargs,
):
    """
    Execute a single problem/algorithm/engine combination.

    Parameters mirror the CLI; additional keyword args are passed to `run_single`
    (e.g., `external_archive_size`, `hv_stop_config`, `track_genealogy`).
    """
    cfg = config or ExperimentConfig()
    selection = make_problem_selection(problem, n_var=n_var, n_obj=n_obj)
    return run_single(
        engine,
        algorithm,
        selection,
        cfg,
        selection_pressure=selection_pressure,
        **kwargs,
    )


def run_experiments_from_args(args: Namespace, config: ExperimentConfig) -> None:
    """
    Entry point used by the CLI to execute one or more runs defined by parsed args.
    """
    run_from_args(args, config)


def run_study(
    tasks: Iterable[StudyTask],
    *,
    config_overrides: dict | None = None,
    mirror_output_roots: Sequence[str] | None = ("results",),
) -> list[StudyResult]:
    runner = StudyRunner(mirror_output_roots=mirror_output_roots)
    return runner.run(list(tasks), config_overrides=config_overrides or {})

