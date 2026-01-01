from __future__ import annotations

import argparse

from .types import SpecDefaults


def add_algorithm_arguments(
    parser: argparse.ArgumentParser,
    *,
    spec_defaults: SpecDefaults,
) -> None:
    """Register algorithm-specific arguments on the parser."""
    nsgaii_defaults = spec_defaults.nsgaii_defaults
    moead_defaults = spec_defaults.moead_defaults
    smsemoa_defaults = spec_defaults.smsemoa_defaults
    nsgaiii_defaults = spec_defaults.nsgaiii_defaults

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
        help="Integer creep step for SMS-EMOA (real encoding).",
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
        help="Mutation probability for NSGA-III.",
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
