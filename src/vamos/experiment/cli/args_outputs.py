from __future__ import annotations

import argparse

from .types import SpecDefaults


def add_output_arguments(
    parser: argparse.ArgumentParser,
    *,
    spec_defaults: SpecDefaults,
) -> None:
    """Register output/visualization arguments on the parser."""
    experiment_defaults = spec_defaults.experiment_defaults

    parser.add_argument(
        "--live-viz",
        action="store_true",
        default=bool(experiment_defaults.get("live_viz", False)),
        help="Enable live/streaming Pareto visualization (matplotlib interactive).",
    )
    parser.add_argument(
        "--live-viz-interval",
        type=int,
        default=experiment_defaults.get("live_viz_interval", 5),
        help="Generations between live visualization updates.",
    )
    parser.add_argument(
        "--live-viz-max-points",
        type=int,
        default=experiment_defaults.get("live_viz_max_points", 1000),
        help="Maximum points to plot in live visualization (subsampled if larger).",
    )
    parser.add_argument(
        "--track-genealogy",
        action="store_true",
        default=bool(experiment_defaults.get("track_genealogy", False)),
        help="Record genealogy/operator stats during NSGA-II runs and save them to results.",
    )
    parser.add_argument(
        "--hv-threshold",
        type=float,
        default=experiment_defaults.get("hv_threshold"),
        help=(
            "Stop runs early once the hypervolume reaches this fraction (0-1) of the reference "
            "front's hypervolume. When omitted, runs use the max evaluation budget."
        ),
    )
    parser.add_argument(
        "--hv-reference-front",
        default=experiment_defaults.get("hv_reference_front"),
        help=("Path to a CSV reference front (two columns) used when --hv-threshold is set. Defaults to built-in fronts for ZDT problems."),
    )
    parser.add_argument(
        "--external-archive-size",
        type=int,
        default=experiment_defaults.get("external_archive_size"),
        help="Size of the optional external archive (applies to NSGA-II only).",
    )
