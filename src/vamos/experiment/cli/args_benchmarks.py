from __future__ import annotations

import argparse

from vamos.foundation.problem.resolver import PROBLEM_SET_PRESETS

from .defaults import spec_default
from .spec_args import add_spec_argument
from .types import SpecDefaults


def add_benchmark_arguments(
    parser: argparse.ArgumentParser,
    *,
    spec_defaults: SpecDefaults,
) -> None:
    """Register benchmark/study-related arguments on the parser."""
    experiment_defaults = spec_defaults.experiment_defaults
    if PROBLEM_SET_PRESETS:
        add_spec_argument(
            parser,
            "--problem-set",
            choices=tuple(PROBLEM_SET_PRESETS.keys()),
            default=spec_default(experiment_defaults, "problem_set", None),
            help=(
                "Run a predefined set of benchmark problems sequentially "
                "(e.g., 'families' runs ZDT1, DTLZ2, and WFG4). Overrides --problem."
            ),
        )
    add_spec_argument(
        parser,
        "--experiment",
        choices=("backends",),
        default=spec_default(experiment_defaults, "experiment", None),
        help="Run a predefined experiment (e.g., compare all backends).",
    )
