from __future__ import annotations

from typing import Any


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


__all__ = [
    "_parse_probability_arg",
    "_parse_positive_float",
    "_normalize_operator_args",
    "collect_nsgaii_variation_args",
    "_collect_generic_variation",
]
