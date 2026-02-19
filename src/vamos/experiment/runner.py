from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vamos.experiment.execution import execute_problem_suite
from vamos.experiment.services.orchestrator import run_single
from vamos.experiment.study.api import run_study
from vamos.foundation.core.experiment_config import ExperimentConfig, resolve_engine
from vamos.foundation.problem.registry import make_problem_selection
from vamos.hooks import LiveVisualization


def run_experiment(
    *,
    algorithm: str,
    problem: str,
    engine: str | None = None,
    config: ExperimentConfig | None = None,
    n_var: int | None = None,
    n_obj: int | None = None,
    selection_pressure: int = 2,
    live_viz_factory: Callable[..., LiveVisualization | None] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Execute a single problem/algorithm/engine combination.

    Parameters mirror the CLI; additional keyword args are passed to `run_single`
    (e.g., `external_archive_size`, `hv_stop_config`, `track_genealogy`).
    """
    cfg = config or ExperimentConfig()
    selection = make_problem_selection(problem, n_var=n_var, n_obj=n_obj)
    live_viz = kwargs.pop("live_viz", None)
    resolved_engine = resolve_engine(engine, algorithm=algorithm)
    if live_viz is None and live_viz_factory is not None:
        live_viz = live_viz_factory(selection, algorithm, resolved_engine, cfg)
    return run_single(
        resolved_engine,
        algorithm,
        selection,
        cfg,
        selection_pressure=selection_pressure,
        live_viz=live_viz,
        **kwargs,
    )


__all__ = [
    "run_single",
    "execute_problem_suite",
    "run_experiment",
    "run_study",
]
