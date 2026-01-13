from __future__ import annotations

import argparse

from vamos.foundation.core.experiment_config import (
    DEFAULT_ALGORITHM,
    DEFAULT_ENGINE,
    DEFAULT_PROBLEM,
    ENABLED_ALGORITHMS,
    EXPERIMENT_BACKENDS,
    EXTERNAL_ALGORITHM_NAMES,
    OPTIONAL_ALGORITHMS,
    ExperimentConfig,
)
from vamos.foundation.problem.registry import available_problem_names

from .defaults import spec_default
from .spec_args import add_spec_argument
from .types import SpecDefaults


def add_core_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_config: ExperimentConfig,
    spec_defaults: SpecDefaults,
) -> None:
    """Register core experiment arguments on the parser."""
    experiment_defaults = spec_defaults.experiment_defaults

    add_spec_argument(
        parser,
        "--algorithm",
        choices=(
            *ENABLED_ALGORITHMS,
            *OPTIONAL_ALGORITHMS,
            *EXTERNAL_ALGORITHM_NAMES,
            "both",
        ),
        default=spec_default(experiment_defaults, "algorithm", DEFAULT_ALGORITHM),
        help=(
            "Algorithm to run (use 'both' to execute the default internal algorithms sequentially; "
            "combine with --include-external to add third-party baselines)."
        ),
    )
    add_spec_argument(
        parser,
        "--engine",
        choices=tuple(EXPERIMENT_BACKENDS),
        default=spec_default(experiment_defaults, "engine", DEFAULT_ENGINE),
        help="Kernel backend to use (default: numpy).",
    )
    add_spec_argument(
        parser,
        "--output-root",
        default=spec_default(experiment_defaults, "output_root", default_config.output_root),
        help="Directory where run artifacts are stored (default: VAMOS_OUTPUT_ROOT or 'results').",
    )
    add_spec_argument(
        parser,
        "--eval-strategy",
        choices=("serial", "multiprocessing"),
        default=spec_default(experiment_defaults, "eval_strategy", "serial"),
        help="Evaluation strategy to use (default: serial).",
    )
    add_spec_argument(
        parser,
        "--n-workers",
        type=int,
        default=experiment_defaults.get("n_workers"),
        help="Number of workers for multiprocessing evaluation backend.",
    )
    add_spec_argument(
        parser,
        "--autodiff-constraints",
        action="store_true",
        default=bool(experiment_defaults.get("autodiff_constraints", False)),
        help="Attempt to build JAX-based constraint evaluators when a ConstraintModel is available.",
    )
    add_spec_argument(
        parser,
        "--problem",
        choices=available_problem_names(),
        default=spec_default(experiment_defaults, "problem", DEFAULT_PROBLEM),
        help="Benchmark problem to solve.",
    )
    add_spec_argument(
        parser,
        "--n-var",
        type=int,
        default=experiment_defaults.get("n_var"),
        help="Override the number of decision variables for the selected problem.",
    )
    add_spec_argument(
        parser,
        "--n-obj",
        type=int,
        default=experiment_defaults.get("n_obj"),
        help="Override the number of objectives (if the problem supports it).",
    )
    add_spec_argument(
        parser,
        "--include-external",
        action="store_true",
        default=bool(spec_default(experiment_defaults, "include_external", False)),
        help="Include PyMOO/jMetalPy/PyGMO baselines when running algorithms.",
    )
    add_spec_argument(
        parser,
        "--external-problem-source",
        choices=("native", "vamos"),
        default=spec_default(experiment_defaults, "external_problem_source", "native"),
        help=(
            "For external baselines, choose whether to use each library's native benchmark "
            "implementation ('native') or wrap the VAMOS problem definition ('vamos')."
        ),
    )
    add_spec_argument(
        parser,
        "--max-evaluations",
        type=int,
        default=spec_default(experiment_defaults, "max_evaluations", default_config.max_evaluations),
        help="Maximum number of evaluations per run (default: %(default)s).",
    )
    add_spec_argument(
        parser,
        "--population-size",
        type=int,
        default=spec_default(experiment_defaults, "population_size", default_config.population_size),
        help="Population size for all internal algorithms (default: %(default)s).",
    )
    add_spec_argument(
        parser,
        "--offspring-population-size",
        type=int,
        default=experiment_defaults.get("offspring_population_size"),
        help=("Number of offspring generated per NSGA-II generation. Defaults to --population-size."),
    )
    add_spec_argument(
        parser,
        "--selection-pressure",
        type=int,
        default=spec_default(experiment_defaults, "selection_pressure", 2),
        help=("Tournament size for selection operators (default: %(default)s). Applies to NSGA-II/III and SMS-EMOA."),
    )
    add_spec_argument(
        parser,
        "--seed",
        type=int,
        default=spec_default(experiment_defaults, "seed", default_config.seed),
        help="Random seed used by internal and external runs (default: %(default)s).",
    )
    add_spec_argument(
        parser,
        "--verbose",
        action="store_true",
        default=bool(experiment_defaults.get("verbose", False)),
        help="Enable verbose logging (DEBUG).",
    )
    add_spec_argument(
        parser,
        "--quiet",
        action="store_true",
        default=bool(experiment_defaults.get("quiet", False)),
        help="Suppress informational logging (only WARNING+).",
    )
