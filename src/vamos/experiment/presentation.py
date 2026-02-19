from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vamos.experiment.runner_utils import run_output_dir
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import ProblemSelection

from .cli.orchestration import run_from_args

if TYPE_CHECKING:
    from vamos.hooks import LiveVisualization


def build_live_viz(
    selection: ProblemSelection,
    algorithm: str,
    engine: str,
    config: ExperimentConfig,
) -> LiveVisualization | None:
    if not getattr(config, "live_viz", False):
        return None
    from vamos.ux.visualization.live_viz import LiveParetoPlot

    output_dir = run_output_dir(selection, algorithm, engine, config.seed, config)
    return LiveParetoPlot(
        update_interval=getattr(config, "live_viz_interval", 5),
        max_points=getattr(config, "live_viz_max_points", 1000),
        save_final_path=os.path.join(output_dir, "live_pareto.png"),
        title=f"{selection.spec.label} (live)",
    )


def resolve_plotter(args: Namespace) -> Callable[..., Any] | None:
    if not getattr(args, "plot", False):
        return None
    from vamos.ux.visualization import plotting

    return plotting.plot_pareto_front


def run_experiments_from_args(args: Namespace, config: ExperimentConfig) -> None:
    """
    CLI entry point that wires execution with optional visualization/presentation.
    """
    plotter = resolve_plotter(args)
    run_from_args(
        args,
        config,
        live_viz_factory=build_live_viz,
        plotter=plotter,
    )


__all__ = ["run_experiments_from_args", "build_live_viz", "resolve_plotter"]
