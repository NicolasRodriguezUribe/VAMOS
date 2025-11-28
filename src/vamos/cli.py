from __future__ import annotations

import argparse

from vamos.problem.registry import available_problem_names
from vamos.runner import (
    DEFAULT_ALGORITHM,
    DEFAULT_ENGINE,
    DEFAULT_PROBLEM,
    ENABLED_ALGORITHMS,
    OPTIONAL_ALGORITHMS,
    EXTERNAL_ALGORITHM_NAMES,
    EXPERIMENT_BACKENDS,
    PROBLEM_SET_PRESETS,
    REFERENCE_FRONT_PATHS,
    ExperimentConfig,
)


def _parse_probability_arg(parser, flag: str, raw, *, allow_expression: bool):
    if raw is None:
        return None
    text = str(raw).strip()
    if allow_expression and text.endswith("/n"):
        numerator = text[:-2].strip()
        if numerator:
            try:
                float(numerator)
            except ValueError:  # pragma: no cover - parser guards
                parser.error(f"{flag} numerator must be numeric; got '{numerator}'.")
            return f"{numerator}/n"
        return "1/n"
    try:
        value = float(text)
    except ValueError:  # pragma: no cover - parser guards
        parser.error(f"{flag} must be a float in [0, 1] or an expression like '1/n'.")
    if not 0.0 <= value <= 1.0:
        parser.error(f"{flag} must be within [0, 1].")
    return value


def _parse_positive_float(parser, flag: str, raw, *, allow_zero: bool):
    if raw is None:
        return None
    value = float(raw)
    if allow_zero:
        if value < 0.0:
            parser.error(f"{flag} must be non-negative.")
    else:
        if value <= 0.0:
            parser.error(f"{flag} must be positive.")
    return value


def _normalize_operator_args(parser, args):
    args.nsgaii_crossover_prob = _parse_probability_arg(
        parser, "--nsgaii-crossover-prob", args.nsgaii_crossover_prob, allow_expression=False
    )
    args.nsgaii_crossover_eta = _parse_positive_float(
        parser, "--nsgaii-crossover-eta", args.nsgaii_crossover_eta, allow_zero=False
    )
    args.nsgaii_crossover_alpha = _parse_positive_float(
        parser, "--nsgaii-crossover-alpha", args.nsgaii_crossover_alpha, allow_zero=True
    )
    args.nsgaii_mutation_prob = _parse_probability_arg(
        parser, "--nsgaii-mutation-prob", args.nsgaii_mutation_prob, allow_expression=True
    )
    args.nsgaii_mutation_eta = _parse_positive_float(
        parser, "--nsgaii-mutation-eta", args.nsgaii_mutation_eta, allow_zero=False
    )
    args.nsgaii_mutation_perturbation = _parse_positive_float(
        parser, "--nsgaii-mutation-perturbation", args.nsgaii_mutation_perturbation, allow_zero=False
    )


def collect_nsgaii_variation_args(args) -> dict:
    return {
        "crossover": {
            "method": getattr(args, "nsgaii_crossover", None),
            "prob": getattr(args, "nsgaii_crossover_prob", None),
            "eta": getattr(args, "nsgaii_crossover_eta", None),
            "alpha": getattr(args, "nsgaii_crossover_alpha", None),
        },
        "mutation": {
            "method": getattr(args, "nsgaii_mutation", None),
            "prob": getattr(args, "nsgaii_mutation_prob", None),
            "eta": getattr(args, "nsgaii_mutation_eta", None),
            "perturbation": getattr(args, "nsgaii_mutation_perturbation", None),
        },
        "repair": getattr(args, "nsgaii_repair", None),
    }


def parse_args(default_config: ExperimentConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectorized multi-objective optimization demo across benchmark problems."
    )
    parser.add_argument(
        "--algorithm",
        choices=(
            *ENABLED_ALGORITHMS,
            *OPTIONAL_ALGORITHMS,
            *EXTERNAL_ALGORITHM_NAMES,
            "both",
        ),
        default=DEFAULT_ALGORITHM,
        help=(
            "Algorithm to run (use 'both' to execute the default internal algorithms sequentially; "
            "combine with --include-external to add third-party baselines)."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=tuple(EXPERIMENT_BACKENDS),
        default=DEFAULT_ENGINE,
        help="Kernel backend to use (default: numpy).",
    )
    parser.add_argument(
        "--problem",
        choices=available_problem_names(),
        default=DEFAULT_PROBLEM,
        help="Benchmark problem to solve.",
    )
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
        "--n-var",
        type=int,
        help="Override the number of decision variables for the selected problem.",
    )
    parser.add_argument(
        "--n-obj",
        type=int,
        help="Override the number of objectives (if the problem supports it).",
    )
    parser.add_argument(
        "--experiment",
        choices=("backends",),
        help="Run a predefined experiment (e.g., compare all backends).",
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Include PyMOO/jMetalPy/PyGMO baselines when running algorithms.",
    )
    parser.add_argument(
        "--external-problem-source",
        choices=("native", "vamos"),
        default="native",
        help=(
            "For external baselines, choose whether to use each library's native benchmark "
            "implementation ('native') or wrap the VAMOS problem definition ('vamos')."
        ),
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=default_config.max_evaluations,
        help="Maximum number of evaluations per run (default: %(default)s).",
    )
    parser.add_argument(
        "--hv-threshold",
        type=float,
        help=(
            "Stop runs early once the hypervolume reaches this fraction (0-1) of the reference "
            "front's hypervolume. When omitted, runs use the max evaluation budget."
        ),
    )
    parser.add_argument(
        "--hv-reference-front",
        help=(
            "Path to a CSV reference front (two columns) used when --hv-threshold is set. "
            "Defaults to built-in fronts for ZDT problems."
        ),
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=default_config.population_size,
        help="Population size for all internal algorithms (default: %(default)s).",
    )
    parser.add_argument(
        "--offspring-population-size",
        type=int,
        help=(
            "Number of offspring generated per NSGA-II generation (must be even). "
            "Defaults to --population-size."
        ),
    )
    parser.add_argument(
        "--selection-pressure",
        type=int,
        default=2,
        help=(
            "Tournament size for selection operators (default: %(default)s). "
            "Applies to NSGA-II/III and SMS-EMOA."
        ),
    )
    parser.add_argument(
        "--nsgaii-crossover",
        choices=(
            "sbx",
            "blx_alpha",
            "ox",
            "order",
            "pmx",
            "cycle",
            "cx",
            "position",
            "position_based",
            "pos",
            "edge",
            "edge_recombination",
            "erx",
            "oxd",
        ),
        default=None,
        help=(
            "Crossover operator for NSGA-II. Continuous problems support sbx/blx_alpha "
            "(default: sbx); permutation problems support ox/pmx/cycle/position/edge "
            "(default: ox)."
        ),
    )
    parser.add_argument(
        "--nsgaii-crossover-prob",
        help="Crossover probability for NSGA-II real-coded operators (default: 0.9).",
    )
    parser.add_argument(
        "--nsgaii-crossover-eta",
        type=float,
        help="Distribution index eta for SBX crossover (default: 20.0).",
    )
    parser.add_argument(
        "--nsgaii-crossover-alpha",
        type=float,
        help="Alpha parameter for BLX-alpha crossover (default: 0.5).",
    )
    parser.add_argument(
        "--nsgaii-mutation",
        choices=(
            "pm",
            "non_uniform",
            "swap",
            "insert",
            "scramble",
            "inversion",
            "simple_inversion",
            "simpleinv",
            "displacement",
        ),
        default=None,
        help=(
            "Mutation operator for NSGA-II. Continuous problems support pm/non_uniform "
            "(default: pm); permutation problems support swap/insert/scramble/"
            "inversion/displacement (default: swap)."
        ),
    )
    parser.add_argument(
        "--nsgaii-mutation-prob",
        help=(
            "Mutation probability for NSGA-II. Accepts floats or expressions "
            "like '1/n' (defaults: 1/n for continuous, 2/n for permutation)."
        ),
    )
    parser.add_argument(
        "--nsgaii-mutation-eta",
        type=float,
        help="Distribution index for polynomial mutation (default: 20.0).",
    )
    parser.add_argument(
        "--nsgaii-mutation-perturbation",
        type=float,
        help="Perturbation strength for non-uniform mutation (default: 0.5).",
    )
    parser.add_argument(
        "--nsgaii-repair",
        choices=("clip", "reflect", "random", "resample", "round", "none"),
        default="clip",
        help=(
            "Repair strategy applied after NSGA-II variation on continuous problems. "
            "'random' and 'resample' behave identically; 'round' snaps to the nearest integer "
            "before clamping. Use 'none' to skip repairs."
        ),
    )
    parser.add_argument(
        "--external-archive-size",
        type=int,
        help=(
            "Enable an external archive for NSGA-II with the given capacity "
            "(ignored by other algorithms)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_config.seed,
        help="Random seed used by internal and external runs (default: %(default)s).",
    )

    args = parser.parse_args()
    _normalize_operator_args(parser, args)
    if args.population_size <= 0:
        parser.error("--population-size must be a positive integer.")
    if args.offspring_population_size is None:
        args.offspring_population_size = args.population_size
    if args.offspring_population_size <= 0:
        parser.error("--offspring-population-size must be positive.")
    if args.offspring_population_size % 2 != 0:
        parser.error("--offspring-population-size must be an even integer.")
    if args.selection_pressure <= 0:
        parser.error("--selection-pressure must be a positive integer.")
    if args.external_archive_size is not None and args.external_archive_size <= 0:
        parser.error("--external-archive-size must be a positive integer.")
    if args.max_evaluations <= 0:
        parser.error("--max-evaluations must be a positive integer.")
    if args.hv_threshold is not None:
        if not (0.0 < args.hv_threshold < 1.0):
            parser.error("--hv-threshold must be in the (0, 1) range.")
        if not args.hv_reference_front:
            default_front = REFERENCE_FRONT_PATHS.get(args.problem.lower())
            if default_front is None:
                parser.error(
                    "--hv-reference-front is required for the selected problem "
                    "because no default reference front is available."
                )
            args.hv_reference_front = str(default_front)
    elif args.hv_reference_front:
        parser.error("--hv-reference-front requires --hv-threshold to be set.")

    args.nsgaii_variation = collect_nsgaii_variation_args(args)
    return args
