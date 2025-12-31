from __future__ import annotations

import argparse

from vamos.foundation.problem.registry import available_problem_names
from vamos.foundation.core.experiment_config import (
    DEFAULT_ALGORITHM,
    DEFAULT_ENGINE,
    DEFAULT_PROBLEM,
    ENABLED_ALGORITHMS,
    OPTIONAL_ALGORITHMS,
    EXTERNAL_ALGORITHM_NAMES,
    EXPERIMENT_BACKENDS,
    ExperimentConfig,
)
from vamos.foundation.problem.resolver import PROBLEM_SET_PRESETS

from .defaults import spec_default
from .types import SpecDefaults


def build_pre_parser() -> argparse.ArgumentParser:
    """Create a pre-parser for config-file discovery."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        help="Path to a YAML/JSON experiment specification. CLI arguments override file values.",
    )
    return pre_parser


def build_parser(
    *,
    default_config: ExperimentConfig,
    pre_parser: argparse.ArgumentParser,
    spec_defaults: SpecDefaults,
) -> argparse.ArgumentParser:
    """Construct the full CLI parser with defaults applied."""
    experiment_defaults = spec_defaults.experiment_defaults
    nsgaii_defaults = spec_defaults.nsgaii_defaults
    moead_defaults = spec_defaults.moead_defaults
    smsemoa_defaults = spec_defaults.smsemoa_defaults
    nsgaiii_defaults = spec_defaults.nsgaiii_defaults

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
        default=spec_default(experiment_defaults, "algorithm", DEFAULT_ALGORITHM),
        help=(
            "Algorithm to run (use 'both' to execute the default internal algorithms sequentially; "
            "combine with --include-external to add third-party baselines)."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=tuple(EXPERIMENT_BACKENDS),
        default=spec_default(experiment_defaults, "engine", DEFAULT_ENGINE),
        help="Kernel backend to use (default: numpy).",
    )
    parser.add_argument(
        "--output-root",
        default=spec_default(experiment_defaults, "output_root", default_config.output_root),
        help="Directory where run artifacts are stored (default: VAMOS_OUTPUT_ROOT or 'results').",
    )
    parser.add_argument(
        "--eval-backend",
        choices=("serial", "multiprocessing"),
        default=spec_default(experiment_defaults, "eval_backend", "serial"),
        help="Evaluation backend to use (default: serial).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=experiment_defaults.get("n_workers"),
        help="Number of workers for multiprocessing evaluation backend.",
    )
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
        "--autodiff-constraints",
        action="store_true",
        default=bool(experiment_defaults.get("autodiff_constraints", False)),
        help="Attempt to build JAX-based constraint evaluators when a ConstraintModel is available.",
    )
    parser.add_argument(
        "--problem",
        choices=available_problem_names(),
        default=spec_default(experiment_defaults, "problem", DEFAULT_PROBLEM),
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
        default=bool(spec_default(experiment_defaults, "include_external", False)),
        help="Include PyMOO/jMetalPy/PyGMO baselines when running algorithms.",
    )
    parser.add_argument(
        "--external-problem-source",
        choices=("native", "vamos"),
        default=spec_default(experiment_defaults, "external_problem_source", "native"),
        help=(
            "For external baselines, choose whether to use each library's native benchmark "
            "implementation ('native') or wrap the VAMOS problem definition ('vamos')."
        ),
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=spec_default(
            experiment_defaults, "max_evaluations", default_config.max_evaluations
        ),
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
        default=spec_default(
            experiment_defaults, "population_size", default_config.population_size
        ),
        help="Population size for all internal algorithms (default: %(default)s).",
    )
    parser.add_argument(
        "--offspring-population-size",
        type=int,
        default=experiment_defaults.get("offspring_population_size"),
        help=(
            "Number of offspring generated per NSGA-II generation. "
            "Defaults to --population-size."
        ),
    )
    parser.add_argument(
        "--selection-pressure",
        type=int,
        default=spec_default(experiment_defaults, "selection_pressure", 2),
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
        default=(nsgaiii_defaults.get("crossover", {}) or {}).get("method"),
        help="Crossover method for NSGA-III (sbx for real, uniform for binary/integer).",
    )
    parser.add_argument(
        "--nsga3-crossover-prob",
        default=(nsgaiii_defaults.get("crossover", {}) or {}).get("prob"),
        help="Crossover probability for NSGA-III.",
    )
    parser.add_argument(
        "--nsga3-crossover-eta",
        type=float,
        default=(nsgaiii_defaults.get("crossover", {}) or {}).get("eta"),
        help="SBX eta for NSGA-III (real encoding).",
    )
    parser.add_argument(
        "--nsga3-mutation",
        choices=("pm", "bitflip", "reset"),
        default=(nsgaiii_defaults.get("mutation", {}) or {}).get("method"),
        help="Mutation method for NSGA-III (pm for real, bitflip for binary, reset for integer).",
    )
    parser.add_argument(
        "--nsga3-mutation-prob",
        default=(nsgaiii_defaults.get("mutation", {}) or {}).get("prob"),
        help="Mutation probability for NSGA-III (allow expressions like 1/n).",
    )
    parser.add_argument(
        "--nsga3-mutation-eta",
        type=float,
        default=(nsgaiii_defaults.get("mutation", {}) or {}).get("eta"),
        help="Polynomial mutation eta for NSGA-III (real encoding).",
    )
    parser.add_argument(
        "--nsga3-mutation-step",
        type=int,
        default=(nsgaiii_defaults.get("mutation", {}) or {}).get("step"),
        help="Integer creep step for NSGA-III (real encoding).",
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
        default=spec_default(experiment_defaults, "seed", default_config.seed),
        help="Random seed used by internal and external runs (default: %(default)s).",
    )

    return parser
