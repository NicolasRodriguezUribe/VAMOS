from __future__ import annotations

import argparse

from vamos.foundation.problem.resolver import PROBLEM_SET_PRESETS

from .types import SpecDefaults


def add_benchmark_arguments(
    parser: argparse.ArgumentParser,
    *,
    spec_defaults: SpecDefaults,
) -> None:
    """Register benchmark/study-related arguments on the parser."""
    _ = spec_defaults
    if PROBLEM_SET_PRESETS:
        parser.add_argument(
            "--problem-set",
            choices=tuple(PROBLEM_SET_PRESETS.keys()),
            help=(
                "Run a predefined set of benchmark problems sequentially "
                "(e.g., 'families' runs ZDT1, DTLZ2, and WFG4). Overrides --problem."
            ),
        )
    parser.add_argument(
        "--experiment",
        choices=("backends",),
        help="Run a predefined experiment (e.g., compare all backends).",
    )
