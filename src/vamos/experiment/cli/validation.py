from __future__ import annotations

import argparse

from vamos.foundation.problem.resolver import resolve_reference_front_path
from vamos.foundation.problem.resolver import PROBLEM_SET_PRESETS

from .common import _normalize_operator_args, collect_nsgaii_variation_args
from .types import SpecDefaults


def finalize_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    spec_defaults: SpecDefaults,
    config_path: str | None,
) -> argparse.Namespace:
    """Normalize and validate parsed CLI arguments."""
    args.config_path = config_path
    args.config_spec = spec_defaults.spec
    args.problem_overrides = spec_defaults.problem_overrides
    args.experiment_defaults = spec_defaults.experiment_defaults

    _normalize_operator_args(parser, args)

    if getattr(args, "quiet", False) and getattr(args, "verbose", False):
        parser.error("--quiet and --verbose cannot be used together.")

    if getattr(args, "problem_set", None) is not None:
        if not PROBLEM_SET_PRESETS:
            parser.error("--problem-set is not available because no presets are registered.")
        if args.problem_set not in PROBLEM_SET_PRESETS:
            parser.error(f"--problem-set must be one of: {', '.join(sorted(PROBLEM_SET_PRESETS))}.")
    if getattr(args, "experiment", None) is not None and args.experiment not in ("backends",):
        parser.error("--experiment must be one of: backends.")

    if args.population_size <= 0:
        parser.error("--population-size must be a positive integer.")
    if args.offspring_population_size is None:
        args.offspring_population_size = args.population_size
    if args.offspring_population_size <= 0:
        parser.error("--offspring-population-size must be positive.")
    if getattr(args, "nsgaii_replacement_size", None) is not None:
        if args.nsgaii_replacement_size <= 0:
            parser.error("--nsgaii-replacement-size must be a positive integer.")
        if args.nsgaii_replacement_size > args.population_size:
            parser.error("--nsgaii-replacement-size must be <= --population-size.")
    if args.selection_pressure <= 0:
        parser.error("--selection-pressure must be a positive integer.")
    if args.external_archive_size is not None and args.external_archive_size <= 0:
        parser.error("--external-archive-size must be a positive integer.")
    if args.max_evaluations <= 0:
        parser.error("--max-evaluations must be a positive integer.")
    if args.n_workers is not None and args.n_workers <= 0:
        parser.error("--n-workers must be positive if provided.")
    if args.live_viz_interval is not None and args.live_viz_interval <= 0:
        parser.error("--live-viz-interval must be positive.")
    if args.hv_threshold is not None:
        if not (0.0 < args.hv_threshold < 1.0):
            parser.error("--hv-threshold must be in the (0, 1) range.")
        if not args.hv_reference_front:
            default_front = resolve_reference_front_path(args.problem.lower(), None)
            if default_front is None:
                parser.error("--hv-reference-front is required for the selected problem because no default reference front is available.")
            args.hv_reference_front = str(default_front)
    elif args.hv_reference_front:
        parser.error("--hv-reference-front requires --hv-threshold to be set.")

    args.nsgaii_variation = collect_nsgaii_variation_args(args)
    args.spea2_variation = spec_defaults.experiment_defaults.get("spea2", {})
    args.ibea_variation = spec_defaults.experiment_defaults.get("ibea", {})
    args.smpso_variation = spec_defaults.experiment_defaults.get("smpso", {})

    return args
