"""
Shared helpers and operator registries for variation pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vamos.foundation.registry import Registry
from vamos.operators.impl.binary import (
    one_point_crossover,
    two_point_crossover,
    uniform_crossover,
    hux_crossover,
    bit_flip_mutation,
)
from vamos.operators.impl.integer import (
    uniform_integer_crossover,
    arithmetic_integer_crossover,
    integer_sbx_crossover,
    random_reset_mutation,
    creep_mutation,
    integer_polynomial_mutation,
)
from vamos.operators.impl.permutation import order_crossover
from vamos.operators.impl.permutation import (
    pmx_crossover,
    cycle_crossover,
    position_based_crossover,
    edge_recombination_crossover,
    swap_mutation,
    insert_mutation,
    scramble_mutation,
    inversion_mutation,
    displacement_mutation,
)
from vamos.operators.impl.mixed import mixed_crossover, mixed_mutation


def resolve_prob_expression(value: float | int | str | None, n_var: int, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, str) and value.endswith("/n"):
        numerator = value[:-2]
        num = float(numerator) if numerator else 1.0
        return min(1.0, max(num, 0.0) / n_var)
    return float(value)


def prepare_mutation_params(
    mut_params: dict[str, float | int | str | None],
    encoding: str,
    n_var: int,
    prob_factor: float | None = None,
) -> dict[str, float | int | str | None]:
    params = dict(mut_params)
    factor = prob_factor if prob_factor is not None else params.get("prob_factor") or params.get("mutation_prob_factor")
    if factor is not None:
        params["prob"] = float(factor) / max(1, n_var)
    elif "prob" in params:
        params["prob"] = resolve_prob_expression(params["prob"], n_var, 1.0)
    else:
        if encoding == "permutation":
            params["prob"] = min(1.0, 2.0 / max(1, n_var))
        else:
            params["prob"] = 1.0 / max(1, n_var)
    return params


if TYPE_CHECKING:
    PERM_CROSSOVER: Registry[Any]
    PERM_MUTATION: Registry[Any]
    BINARY_CROSSOVER: Registry[Any]
    BINARY_MUTATION: Registry[Any]
    INT_CROSSOVER: Registry[Any]
    INT_MUTATION: Registry[Any]
    MIXED_CROSSOVER: Registry[Any]
    MIXED_MUTATION: Registry[Any]
    REAL_CROSSOVER: Registry[Any]
    REAL_MUTATION: Registry[Any]

_REGISTRY_LABELS = {
    "PERM_CROSSOVER": "Permutation Crossover",
    "PERM_MUTATION": "Permutation Mutation",
    "BINARY_CROSSOVER": "Binary Crossover",
    "BINARY_MUTATION": "Binary Mutation",
    "INT_CROSSOVER": "Integer Crossover",
    "INT_MUTATION": "Integer Mutation",
    "MIXED_CROSSOVER": "Mixed Crossover",
    "MIXED_MUTATION": "Mixed Mutation",
    "REAL_CROSSOVER": "Real Crossover",
    "REAL_MUTATION": "Real Mutation",
}
_REGISTRIES: dict[str, Registry[Any]] = {}
_DEFAULTS_POPULATED = False


def _get_registry(name: str) -> Registry[Any]:
    registry = _REGISTRIES.get(name)
    if registry is None:
        registry = Registry(_REGISTRY_LABELS[name])
        _REGISTRIES[name] = registry
    return registry


def _populate(reg: Registry[Any], items: dict[str, object]) -> None:
    for key, value in items.items():
        reg.register(key, value)


def _populate_defaults() -> None:
    global _DEFAULTS_POPULATED
    if _DEFAULTS_POPULATED:
        return
    perm_crossover = _get_registry("PERM_CROSSOVER")
    if len(perm_crossover) > 0:
        _DEFAULTS_POPULATED = True
        return

    perm_mutation = _get_registry("PERM_MUTATION")
    binary_crossover = _get_registry("BINARY_CROSSOVER")
    binary_mutation = _get_registry("BINARY_MUTATION")
    int_crossover = _get_registry("INT_CROSSOVER")
    int_mutation = _get_registry("INT_MUTATION")
    mixed_crossover_registry = _get_registry("MIXED_CROSSOVER")
    mixed_mutation_registry = _get_registry("MIXED_MUTATION")
    real_crossover = _get_registry("REAL_CROSSOVER")
    real_mutation = _get_registry("REAL_MUTATION")

    _populate(
        perm_crossover,
        {
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
        },
    )

    _populate(
        perm_mutation,
        {
            "swap": swap_mutation,
            "insert": insert_mutation,
            "scramble": scramble_mutation,
            "inversion": inversion_mutation,
            "displacement": displacement_mutation,
        },
    )

    _populate(
        binary_crossover,
        {
            "one_point": one_point_crossover,
            "single_point": one_point_crossover,
            "1point": one_point_crossover,
            "spx": one_point_crossover,
            "two_point": two_point_crossover,
            "2point": two_point_crossover,
            "uniform": uniform_crossover,
            "hux": hux_crossover,
        },
    )

    _populate(
        binary_mutation,
        {
            "bitflip": bit_flip_mutation,
            "bit_flip": bit_flip_mutation,
        },
    )

    _populate(
        int_crossover,
        {
            "uniform": uniform_integer_crossover,
            "blend": arithmetic_integer_crossover,
            "arithmetic": arithmetic_integer_crossover,
            "sbx": integer_sbx_crossover,
        },
    )

    _populate(
        int_mutation,
        {
            "reset": random_reset_mutation,
            "random_reset": random_reset_mutation,
            "creep": creep_mutation,
            "pm": integer_polynomial_mutation,
            "polynomial": integer_polynomial_mutation,
        },
    )

    _populate(
        mixed_crossover_registry,
        {
            "mixed": mixed_crossover,
            "uniform": mixed_crossover,
        },
    )

    _populate(
        mixed_mutation_registry,
        {
            "mixed": mixed_mutation,
            "gaussian": mixed_mutation,
        },
    )

    _populate(real_crossover, {k: "PLACEHOLDER" for k in ["sbx", "blx_alpha", "arithmetic", "pcx", "undx", "simplex"]})

    _populate(
        real_mutation,
        {k: "PLACEHOLDER" for k in ["pm", "non_uniform", "gaussian", "uniform_reset", "cauchy", "uniform", "linked_polynomial"]},
    )
    _DEFAULTS_POPULATED = True


def validate_operator_support(encoding: str, crossover: str, mutation: str) -> None:
    _populate_defaults()
    perm_crossover = _get_registry("PERM_CROSSOVER")
    perm_mutation = _get_registry("PERM_MUTATION")
    binary_crossover = _get_registry("BINARY_CROSSOVER")
    binary_mutation = _get_registry("BINARY_MUTATION")
    int_crossover = _get_registry("INT_CROSSOVER")
    int_mutation = _get_registry("INT_MUTATION")
    mixed_crossover = _get_registry("MIXED_CROSSOVER")
    mixed_mutation = _get_registry("MIXED_MUTATION")
    real_crossover = _get_registry("REAL_CROSSOVER")
    real_mutation = _get_registry("REAL_MUTATION")
    if encoding == "permutation":
        if crossover not in perm_crossover:
            raise ValueError(f"Unsupported crossover '{crossover}' for permutation encoding.")
        if mutation not in perm_mutation:
            raise ValueError(f"Unsupported mutation '{mutation}' for permutation encoding.")
    elif encoding == "binary":
        if crossover not in binary_crossover:
            raise ValueError(f"Unsupported crossover '{crossover}' for binary encoding.")
        if mutation not in binary_mutation:
            raise ValueError(f"Unsupported mutation '{mutation}' for binary encoding.")
    elif encoding == "integer":
        if crossover not in int_crossover:
            raise ValueError(f"Unsupported crossover '{crossover}' for integer encoding.")
        if mutation not in int_mutation:
            raise ValueError(f"Unsupported mutation '{mutation}' for integer encoding.")
    elif encoding == "mixed":
        if crossover not in mixed_crossover:
            raise ValueError(f"Unsupported crossover '{crossover}' for mixed encoding.")
        if mutation not in mixed_mutation:
            raise ValueError(f"Unsupported mutation '{mutation}' for mixed encoding.")
    else:
        if crossover not in real_crossover:
            raise ValueError(f"Unsupported crossover '{crossover}' for continuous encoding.")
        if mutation not in real_mutation:
            raise ValueError(f"Unsupported mutation '{mutation}' for continuous encoding.")


def __getattr__(name: str) -> Registry[Any]:
    if name in _REGISTRY_LABELS:
        _populate_defaults()
        return _get_registry(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
