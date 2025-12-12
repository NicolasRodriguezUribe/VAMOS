"""
Shared helpers and operator registries for variation pipelines.
"""
from __future__ import annotations

import numpy as np

from vamos.operators.binary import (
    one_point_crossover,
    two_point_crossover,
    uniform_crossover,
    hux_crossover,
    bit_flip_mutation,
)
from vamos.operators.integer import (
    uniform_integer_crossover,
    arithmetic_integer_crossover,
    random_reset_mutation,
    creep_mutation,
)
from vamos.operators.permutation import (
    order_crossover,
    pmx_crossover,
    cycle_crossover,
    position_based_crossover,
    edge_recombination_crossover,
    swap_mutation,
    insert_mutation,
    scramble_mutation,
    inversion_mutation,
    simple_inversion_mutation,
    displacement_mutation,
)
from vamos.operators.mixed import mixed_crossover, mixed_mutation


def resolve_prob_expression(value, n_var: int, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, str) and value.endswith("/n"):
        numerator = value[:-2]
        num = float(numerator) if numerator else 1.0
        return min(1.0, max(num, 0.0) / n_var)
    return float(value)


def prepare_mutation_params(mut_params: dict, encoding: str, n_var: int, prob_factor: float | None = None) -> dict:
    params = dict(mut_params)
    factor = prob_factor if prob_factor is not None else params.get("prob_factor") or params.get("mutation_prob_factor")
    if factor is not None:
        params["prob"] = float(factor) / max(1, n_var)
    elif "prob" in params:
        params["prob"] = resolve_prob_expression(params["prob"], n_var, params["prob"])
    else:
        if encoding == "permutation":
            params["prob"] = min(1.0, 2.0 / max(1, n_var))
        else:
            params["prob"] = 1.0 / max(1, n_var)
    return params


PERM_CROSSOVER = {
    "ox": order_crossover,
    "order": order_crossover,
    "oxd": order_crossover,
    "pmx": pmx_crossover,
    "cycle": cycle_crossover,
    "cx": cycle_crossover,
    "position": position_based_crossover,
    "position_based": position_based_crossover,
    "pos": position_based_crossover,
    "edge": edge_recombination_crossover,
    "edge_recombination": edge_recombination_crossover,
    "erx": edge_recombination_crossover,
}

PERM_MUTATION = {
    "swap": swap_mutation,
    "insert": insert_mutation,
    "scramble": scramble_mutation,
    "inversion": inversion_mutation,
    "simple_inversion": simple_inversion_mutation,
    "simpleinv": simple_inversion_mutation,
    "displacement": displacement_mutation,
}

BINARY_CROSSOVER = {
    "one_point": one_point_crossover,
    "single_point": one_point_crossover,
    "1point": one_point_crossover,
    "two_point": two_point_crossover,
    "2point": two_point_crossover,
    "uniform": uniform_crossover,
    "hux": hux_crossover,
}

BINARY_MUTATION = {
    "bitflip": bit_flip_mutation,
    "bit_flip": bit_flip_mutation,
}

INT_CROSSOVER = {
    "uniform": uniform_integer_crossover,
    "blend": arithmetic_integer_crossover,
    "arithmetic": arithmetic_integer_crossover,
}

INT_MUTATION = {
    "reset": random_reset_mutation,
    "random_reset": random_reset_mutation,
    "creep": creep_mutation,
}

MIXED_CROSSOVER = {
    "mixed": mixed_crossover,
    "uniform": mixed_crossover,
}

MIXED_MUTATION = {
    "mixed": mixed_mutation,
    "gaussian": mixed_mutation,
}


def validate_operator_support(encoding: str, crossover: str, mutation: str) -> None:
    if encoding == "permutation":
        if crossover not in PERM_CROSSOVER:
            raise ValueError(f"Unsupported crossover '{crossover}' for permutation encoding.")
        if mutation not in PERM_MUTATION:
            raise ValueError(f"Unsupported mutation '{mutation}' for permutation encoding.")
    elif encoding == "binary":
        if crossover not in BINARY_CROSSOVER:
            raise ValueError(f"Unsupported crossover '{crossover}' for binary encoding.")
        if mutation not in BINARY_MUTATION:
            raise ValueError(f"Unsupported mutation '{mutation}' for binary encoding.")
    elif encoding == "integer":
        if crossover not in INT_CROSSOVER:
            raise ValueError(f"Unsupported crossover '{crossover}' for integer encoding.")
        if mutation not in INT_MUTATION:
            raise ValueError(f"Unsupported mutation '{mutation}' for integer encoding.")
    elif encoding == "mixed":
        if crossover not in MIXED_CROSSOVER:
            raise ValueError(f"Unsupported crossover '{crossover}' for mixed encoding.")
        if mutation not in MIXED_MUTATION:
            raise ValueError(f"Unsupported mutation '{mutation}' for mixed encoding.")
    else:
        if crossover not in {"sbx", "blx_alpha", "arithmetic", "pcx", "undx", "spx"}:
            raise ValueError(f"Unsupported crossover '{crossover}' for continuous encoding.")
        if mutation not in {"pm", "non_uniform", "gaussian", "uniform_reset", "cauchy", "uniform", "linked_polynomial"}:
            raise ValueError(f"Unsupported mutation '{mutation}' for continuous encoding.")


__all__ = [
    "resolve_prob_expression",
    "prepare_mutation_params",
    "validate_operator_support",
    "PERM_CROSSOVER",
    "PERM_MUTATION",
    "BINARY_CROSSOVER",
    "BINARY_MUTATION",
    "INT_CROSSOVER",
    "INT_MUTATION",
    "MIXED_CROSSOVER",
    "MIXED_MUTATION",
]
