from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

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


def _load_experiment_spec(path: str) -> Dict[str, Any]:
    spec_path = Path(path).expanduser().resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"Config file '{spec_path}' does not exist.")
    suffix = spec_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "YAML config requested but PyYAML is not installed. Install with 'pip install pyyaml'."
            ) from exc
        with spec_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    with spec_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


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
    args.moead_mutation_prob = _parse_probability_arg(
        parser, "--moead-mutation-prob", getattr(args, "moead_mutation_prob", None), allow_expression=True
    )
    args.moead_crossover_prob = _parse_probability_arg(
        parser, "--moead-crossover-prob", getattr(args, "moead_crossover_prob", None), allow_expression=False
    )
    args.smsemoa_mutation_prob = _parse_probability_arg(
        parser, "--smsemoa-mutation-prob", getattr(args, "smsemoa_mutation_prob", None), allow_expression=True
    )
    args.smsemoa_crossover_prob = _parse_probability_arg(
        parser, "--smsemoa-crossover-prob", getattr(args, "smsemoa_crossover_prob", None), allow_expression=False
    )
    args.nsga3_mutation_prob = _parse_probability_arg(
        parser, "--nsga3-mutation-prob", getattr(args, "nsga3_mutation_prob", None), allow_expression=True
    )
    args.nsga3_crossover_prob = _parse_probability_arg(
        parser, "--nsga3-crossover-prob", getattr(args, "nsga3_crossover_prob", None), allow_expression=False
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


def _collect_generic_variation(args, prefix: str) -> dict:
    return {
        "crossover": {
            "method": getattr(args, f"{prefix}_crossover", None),
            "prob": getattr(args, f"{prefix}_crossover_prob", None),
            "eta": getattr(args, f"{prefix}_crossover_eta", None),
        },
        "mutation": {
            "method": getattr(args, f"{prefix}_mutation", None),
            "prob": getattr(args, f"{prefix}_mutation_prob", None),
            "eta": getattr(args, f"{prefix}_mutation_eta", None),
            "perturbation": getattr(args, f"{prefix}_mutation_perturbation", None),
            "step": getattr(args, f"{prefix}_mutation_step", None),
        },
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
    args.moead_variation = _collect_generic_variation(args, "moead")
    args.smsemoa_variation = _collect_generic_variation(args, "smsemoa")
    args.nsga3_variation = _collect_generic_variation(args, "nsga3")
    return args


# Config-file aware parser (overrides the legacy definition above).
def parse_args(default_config: ExperimentConfig) -> argparse.Namespace:  # type: ignore[override]
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        help="Path to a YAML/JSON experiment specification. CLI arguments override file values.",
    )
    pre_args, remaining = pre_parser.parse_known_args()

    spec = {}
    problem_overrides = {}
    experiment_defaults: dict[str, Any] = {}
    nsgaii_defaults: dict[str, Any] = {}
    moead_defaults: dict[str, Any] = {}
    smsemoa_defaults: dict[str, Any] = {}
    nsga3_defaults: dict[str, Any] = {}
    if pre_args.config:
        spec = _load_experiment_spec(pre_args.config)
        problem_overrides = spec.get("problems", {}) or {}
        experiment_defaults = spec.get("defaults", {}) or {k: v for k, v in spec.items() if k != "problems"}
        nsgaii_defaults = experiment_defaults.get("nsgaii", {}) or {}
        moead_defaults = experiment_defaults.get("moead", {}) or {}
        smsemoa_defaults = experiment_defaults.get("smsemoa", {}) or {}
        nsga3_defaults = experiment_defaults.get("nsga3", {}) or {}

    def _spec_default(key: str, fallback):
        return experiment_defaults.get(key, fallback)

    parser = argparse.ArgumentParser(
        description="Vectorized multi-objective optimization demo across benchmark problems.",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--algorithm",
        choices=(
            *ENABLED_ALGORITHMS,
            *OPTIONAL_ALGORITHMS,
            *EXTERNAL_ALGORITHM_NAMES,
            "both",
        ),
        default=_spec_default("algorithm", DEFAULT_ALGORITHM),
        help=(
            "Algorithm to run (use 'both' to execute the default internal algorithms sequentially; "
            "combine with --include-external to add third-party baselines)."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=tuple(EXPERIMENT_BACKENDS),
        default=_spec_default("engine", DEFAULT_ENGINE),
        help="Kernel backend to use (default: numpy).",
    )
    parser.add_argument(
        "--problem",
        choices=available_problem_names(),
        default=_spec_default("problem", DEFAULT_PROBLEM),
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
        default=experiment_defaults.get("n_var"),
        help="Override the number of decision variables for the selected problem.",
    )
    parser.add_argument(
        "--n-obj",
        type=int,
        default=experiment_defaults.get("n_obj"),
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
        default=bool(_spec_default("include_external", False)),
        help="Include PyMOO/jMetalPy/PyGMO baselines when running algorithms.",
    )
    parser.add_argument(
        "--external-problem-source",
        choices=("native", "vamos"),
        default=_spec_default("external_problem_source", "native"),
        help=(
            "For external baselines, choose whether to use each library's native benchmark "
            "implementation ('native') or wrap the VAMOS problem definition ('vamos')."
        ),
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=_spec_default("max_evaluations", default_config.max_evaluations),
        help="Maximum number of evaluations per run (default: %(default)s).",
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
        help=(
            "Path to a CSV reference front (two columns) used when --hv-threshold is set. "
            "Defaults to built-in fronts for ZDT problems."
        ),
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=_spec_default("population_size", default_config.population_size),
        help="Population size for all internal algorithms (default: %(default)s).",
    )
    parser.add_argument(
        "--offspring-population-size",
        type=int,
        default=experiment_defaults.get("offspring_population_size"),
        help=(
            "Number of offspring generated per NSGA-II generation (must be even). "
            "Defaults to --population-size."
        ),
    )
    parser.add_argument(
        "--selection-pressure",
        type=int,
        default=_spec_default("selection_pressure", 2),
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
        default=(nsgaii_defaults.get("crossover", {}) or {}).get("method"),
        help=(
            "Crossover operator for NSGA-II. Continuous problems support sbx/blx_alpha "
            "(default: sbx); permutation problems support ox/pmx/cycle/position/edge "
            "(default: ox)."
        ),
    )
    parser.add_argument(
        "--nsgaii-crossover-prob",
        default=(nsgaii_defaults.get("crossover", {}) or {}).get("prob"),
        help="Crossover probability for NSGA-II real-coded operators (default: 0.9).",
    )
    parser.add_argument(
        "--nsgaii-crossover-eta",
        type=float,
        default=(nsgaii_defaults.get("crossover", {}) or {}).get("eta"),
        help="Distribution index eta for SBX crossover (default: 20.0).",
    )
    parser.add_argument(
        "--nsgaii-crossover-alpha",
        type=float,
        default=(nsgaii_defaults.get("crossover", {}) or {}).get("alpha"),
        help="Alpha for BLX-alpha crossover (default: 0.5).",
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
        default=(nsgaii_defaults.get("mutation", {}) or {}).get("method"),
        help=(
            "Mutation operator for NSGA-II. Continuous problems support pm/non_uniform "
            "(default: pm); permutation problems support swap/insert/scramble/inversion/"
            "simple_inversion/displacement (default: swap)."
        ),
    )
    parser.add_argument(
        "--nsgaii-mutation-prob",
        type=str,
        default=(nsgaii_defaults.get("mutation", {}) or {}).get("prob"),
        help="Mutation probability for NSGA-II operators (allow expressions like 1/n).",
    )
    parser.add_argument(
        "--nsgaii-mutation-eta",
        type=float,
        default=(nsgaii_defaults.get("mutation", {}) or {}).get("eta"),
        help="Distribution index eta for polynomial mutation (default: 20.0).",
    )
    parser.add_argument(
        "--nsgaii-mutation-perturbation",
        type=float,
        default=(nsgaii_defaults.get("mutation", {}) or {}).get("perturbation"),
        help="Perturbation magnitude for non-uniform mutation (default: 0.5).",
    )
    parser.add_argument(
        "--nsgaii-repair",
        choices=("clip", "reflect", "random", "resample", "round", "none"),
        default=nsgaii_defaults.get("repair"),
        help="Repair strategy for NSGA-II (continuous encoding).",
    )
    parser.add_argument(
        "--moead-crossover",
        choices=("sbx", "uniform"),
        default=(moead_defaults.get("crossover", {}) or {}).get("method"),
        help="Crossover method for MOEA/D (sbx for real, uniform for binary/integer).",
    )
    parser.add_argument(
        "--moead-crossover-prob",
        default=(moead_defaults.get("crossover", {}) or {}).get("prob"),
        help="Crossover probability for MOEA/D.",
    )
    parser.add_argument(
        "--moead-crossover-eta",
        type=float,
        default=(moead_defaults.get("crossover", {}) or {}).get("eta"),
        help="SBX eta for MOEA/D (real encoding).",
    )
    parser.add_argument(
        "--moead-mutation",
        choices=("pm", "bitflip", "reset"),
        default=(moead_defaults.get("mutation", {}) or {}).get("method"),
        help="Mutation method for MOEA/D (pm for real, bitflip for binary, reset for integer).",
    )
    parser.add_argument(
        "--moead-mutation-prob",
        default=(moead_defaults.get("mutation", {}) or {}).get("prob"),
        help="Mutation probability for MOEA/D (allow expressions like 1/n).",
    )
    parser.add_argument(
        "--moead-mutation-eta",
        type=float,
        default=(moead_defaults.get("mutation", {}) or {}).get("eta"),
        help="Polynomial mutation eta for MOEA/D (real encoding).",
    )
    parser.add_argument(
        "--moead-mutation-step",
        type=int,
        default=(moead_defaults.get("mutation", {}) or {}).get("step"),
        help="Integer creep step for MOEA/D (integer encoding).",
    )
    parser.add_argument(
        "--moead-aggregation",
        default=moead_defaults.get("aggregation"),
        help="Aggregation method for MOEA/D (e.g., tchebycheff, weighted_sum, pbi).",
    )
    parser.add_argument(
        "--smsemoa-crossover",
        choices=("sbx", "uniform"),
        default=(smsemoa_defaults.get("crossover", {}) or {}).get("method"),
        help="Crossover method for SMS-EMOA (sbx for real, uniform for binary/integer).",
    )
    parser.add_argument(
        "--smsemoa-crossover-prob",
        default=(smsemoa_defaults.get("crossover", {}) or {}).get("prob"),
        help="Crossover probability for SMS-EMOA.",
    )
    parser.add_argument(
        "--smsemoa-crossover-eta",
        type=float,
        default=(smsemoa_defaults.get("crossover", {}) or {}).get("eta"),
        help="SBX eta for SMS-EMOA (real encoding).",
    )
    parser.add_argument(
        "--smsemoa-mutation",
        choices=("pm", "bitflip", "reset"),
        default=(smsemoa_defaults.get("mutation", {}) or {}).get("method"),
        help="Mutation method for SMS-EMOA (pm for real, bitflip for binary, reset for integer).",
    )
    parser.add_argument(
        "--smsemoa-mutation-prob",
        default=(smsemoa_defaults.get("mutation", {}) or {}).get("prob"),
        help="Mutation probability for SMS-EMOA (allow expressions like 1/n).",
    )
    parser.add_argument(
        "--smsemoa-mutation-eta",
        type=float,
        default=(smsemoa_defaults.get("mutation", {}) or {}).get("eta"),
        help="Polynomial mutation eta for SMS-EMOA (real encoding).",
    )
    parser.add_argument(
        "--smsemoa-mutation-step",
        type=int,
        default=(smsemoa_defaults.get("mutation", {}) or {}).get("step"),
        help="Integer creep step for SMS-EMOA (integer encoding).",
    )
    parser.add_argument(
        "--nsga3-crossover",
        choices=("sbx", "uniform"),
        default=(nsga3_defaults.get("crossover", {}) or {}).get("method"),
        help="Crossover method for NSGA-III (sbx for real, uniform for binary/integer).",
    )
    parser.add_argument(
        "--nsga3-crossover-prob",
        default=(nsga3_defaults.get("crossover", {}) or {}).get("prob"),
        help="Crossover probability for NSGA-III.",
    )
    parser.add_argument(
        "--nsga3-crossover-eta",
        type=float,
        default=(nsga3_defaults.get("crossover", {}) or {}).get("eta"),
        help="SBX eta for NSGA-III (real encoding).",
    )
    parser.add_argument(
        "--nsga3-mutation",
        choices=("pm", "bitflip", "reset"),
        default=(nsga3_defaults.get("mutation", {}) or {}).get("method"),
        help="Mutation method for NSGA-III (pm for real, bitflip for binary, reset for integer).",
    )
    parser.add_argument(
        "--nsga3-mutation-prob",
        default=(nsga3_defaults.get("mutation", {}) or {}).get("prob"),
        help="Mutation probability for NSGA-III (allow expressions like 1/n).",
    )
    parser.add_argument(
        "--nsga3-mutation-eta",
        type=float,
        default=(nsga3_defaults.get("mutation", {}) or {}).get("eta"),
        help="Polynomial mutation eta for NSGA-III (real encoding).",
    )
    parser.add_argument(
        "--nsga3-mutation-step",
        type=int,
        default=(nsga3_defaults.get("mutation", {}) or {}).get("step"),
        help="Integer creep step for NSGA-III (integer encoding).",
    )
    parser.add_argument(
        "--external-archive-size",
        type=int,
        default=experiment_defaults.get("external_archive_size"),
        help="Size of the optional external archive (applies to NSGA-II only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_spec_default("seed", default_config.seed),
        help="Random seed used by internal and external runs (default: %(default)s).",
    )

    args = parser.parse_args(remaining)
    args.config_path = getattr(pre_args, "config", None)
    args.problem_overrides = problem_overrides
    args.experiment_defaults = experiment_defaults
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
