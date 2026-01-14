"""
Shared helpers for variation/default handling across CLI, runner, and factories.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict, TypeAlias

from vamos.foundation.encoding import normalize_encoding


# =============================================================================
# Operator Registry by Encoding
# =============================================================================

OperatorParams: TypeAlias = dict[str, Any]
OperatorTuple: TypeAlias = tuple[str, OperatorParams]


class OperatorSpecDict(TypedDict, total=False):
    method: str
    name: str


OperatorSpecInput: TypeAlias = str | OperatorTuple | OperatorSpecDict | Mapping[str, object]


class VariationOverrides(TypedDict, total=False):
    crossover: OperatorSpecInput
    mutation: OperatorSpecInput
    selection: OperatorSpecInput
    repair: OperatorSpecInput
    aggregation: OperatorSpecInput
    adaptive_operator_selection: Mapping[str, object]
    weight_vectors: Mapping[str, object] | str
    archive_size: int
    k_neighbors: int
    indicator: str
    kappa: float
    inertia: float
    c1: float
    c2: float
    vmax_fraction: float
    n_partitions: int
    alpha: float
    adapt_freq: float


VariationConfig: TypeAlias = dict[str, object]

OPERATORS_BY_ENCODING: dict[str, dict[str, list[OperatorTuple]]] = {
    "real": {
        "crossover": [
            ("sbx", {"prob": 1.0, "eta": 20.0}),
            ("arithmetic", {"prob": 0.9}),
            ("blx_alpha", {"prob": 0.9, "alpha": 0.5}),
            ("pcx", {"prob": 0.9}),
            ("undx", {"prob": 0.9}),
            ("simplex", {"prob": 0.9}),
        ],
        "mutation": [
            ("pm", {"prob": "1/n", "eta": 20.0}),
            ("gaussian", {"prob": 0.1, "sigma": 0.1}),
            ("uniform_reset", {"prob": 0.1}),
        ],
    },
    "binary": {
        "crossover": [
            ("hux", {"prob": 0.9}),
            ("uniform", {"prob": 0.9}),
            ("one_point", {"prob": 0.9}),
            ("two_point", {"prob": 0.9}),
        ],
        "mutation": [
            ("bitflip", {"prob": "1/n"}),
        ],
    },
    "integer": {
        "crossover": [
            ("uniform", {"prob": 0.9}),
            ("arithmetic", {"prob": 0.9}),
        ],
        "mutation": [
            ("reset", {"prob": "1/n"}),
            ("creep", {"prob": 0.1}),
        ],
    },
    "permutation": {
        "crossover": [
            ("ox", {"prob": 0.9}),
            ("pmx", {"prob": 0.9}),
            ("edge", {"prob": 0.9}),
            ("cycle", {"prob": 0.9}),
            ("position", {"prob": 0.9}),
        ],
        "mutation": [
            ("swap", {"prob": 0.1}),
            ("inversion", {"prob": 0.1}),
            ("scramble", {"prob": 0.1}),
            ("insert", {"prob": 0.1}),
            ("displacement", {"prob": 0.1}),
        ],
    },
    "mixed": {
        "crossover": [
            ("mixed", {"prob": 0.9}),
        ],
        "mutation": [
            ("mixed", {"prob": "1/n"}),
        ],
    },
}


def get_operators_for_encoding(encoding: str) -> dict[str, list[OperatorTuple]]:
    """Return available operators for the given encoding type."""
    normalized = normalize_encoding(encoding)
    return OPERATORS_BY_ENCODING.get(normalized, OPERATORS_BY_ENCODING["real"])


def get_crossover_names(encoding: str) -> list[str]:
    """Return list of crossover operator names for given encoding."""
    ops = get_operators_for_encoding(encoding)
    return [op[0] for op in ops.get("crossover", [])]


def get_mutation_names(encoding: str) -> list[str]:
    """Return list of mutation operator names for given encoding."""
    ops = get_operators_for_encoding(encoding)
    return [op[0] for op in ops.get("mutation", [])]


def normalize_operator_tuple(spec: object) -> OperatorTuple | None:
    """
    Accept operator specs in tuple/dict/string form and normalize to (name, params).
    """
    if spec is None:
        return None
    if isinstance(spec, (tuple, list)):
        if len(spec) != 2:
            return None
        name, params_raw = spec
        if isinstance(params_raw, Mapping):
            return (str(name), dict(params_raw))
        return (str(name), {})
    if isinstance(spec, str):
        return (spec, {})
    if isinstance(spec, dict):
        method = spec.get("method") or spec.get("name")
        if not method:
            return None
        op_params: OperatorParams = {str(k): v for k, v in spec.items() if k not in {"method", "name"} and v is not None}
        return (str(method), op_params)
    return None


def ensure_operator_tuple(spec: object, *, key: str) -> OperatorTuple:
    op = normalize_operator_tuple(spec)
    if op is None:
        raise ValueError(f"Invalid operator spec for '{key}': {spec!r}")
    return op


def ensure_operator_tuple_optional(spec: object, *, key: str) -> OperatorTuple | None:
    if spec is None:
        return None
    return ensure_operator_tuple(spec, key=key)


def normalize_variation_config(raw: Mapping[str, object] | None) -> VariationConfig | None:
    """
    Normalize variation configuration dictionaries into operator tuples where applicable.
    """
    if not raw:
        return None
    normalized: VariationConfig = {}
    known_op_keys = {"crossover", "mutation", "selection", "repair", "aggregation"}
    for key in known_op_keys:
        op = normalize_operator_tuple(raw.get(key))
        if op:
            normalized[key] = op
    for key, value in raw.items():
        if key in normalized or key in known_op_keys:
            continue
        if value is not None:
            normalized[key] = value
    return normalized or None


def resolve_default_variation_config(encoding: str, overrides: Mapping[str, object] | None) -> VariationConfig:
    """
    Default variation configuration by encoding, merged with user overrides.
    """
    normalized = normalize_encoding(encoding)
    base: VariationConfig
    if normalized == "real":
        base = {
            "crossover": ("sbx", {"prob": 1.0, "eta": 20.0}),
            "mutation": ("pm", {"prob": "1/n", "eta": 20.0}),
        }
    elif normalized == "binary":
        base = {
            "crossover": ("hux", {"prob": 0.9}),
            "mutation": ("bitflip", {"prob": "1/n"}),
        }
    elif normalized == "integer":
        base = {
            "crossover": ("uniform", {"prob": 0.9}),
            "mutation": ("reset", {"prob": "1/n"}),
        }
    elif normalized == "mixed":
        base = {
            "crossover": ("mixed", {"prob": 0.9}),
            "mutation": ("mixed", {"prob": "1/n"}),
        }
    elif normalized == "permutation":
        base = {
            "crossover": ("ox", {"prob": 0.9}),
            "mutation": ("swap", {"prob": 0.1}),
        }
    else:
        base = {
            "crossover": ("sbx", {"prob": 0.9, "eta": 20.0}),
            "mutation": ("pm", {"prob": 0.1, "eta": 20.0}),
        }

    if overrides:
        if "crossover" in overrides:
            base["crossover"] = overrides["crossover"]
        if "mutation" in overrides:
            base["mutation"] = overrides["mutation"]
        if "repair" in overrides:
            base["repair"] = overrides["repair"]
        for key, value in overrides.items():
            if key in {"crossover", "mutation", "repair"}:
                continue
            base[key] = value

    return base


def merge_variation_overrides(base: Mapping[str, object] | None, override: Mapping[str, object] | None) -> VariationConfig:
    base = base or {}
    if not override:
        return dict(base)
    merged = dict(base)
    merged.update({k: v for k, v in override.items() if v is not None})
    return merged
