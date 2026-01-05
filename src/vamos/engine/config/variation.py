"""
Shared helpers for variation/default handling across CLI, runner, and factories.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


# =============================================================================
# Operator Registry by Encoding
# =============================================================================

OPERATORS_BY_ENCODING: Dict[str, Dict[str, List[Tuple[str, Dict[str, Any]]]]] = {
    "real": {
        "crossover": [
            ("sbx", {"prob": 0.9, "eta": 20.0}),
            ("uniform", {"prob": 0.9}),
            ("blx", {"prob": 0.9, "alpha": 0.5}),
            ("arithmetic", {"prob": 0.9}),
            ("de", {"prob": 0.9, "F": 0.5}),
        ],
        "mutation": [
            ("pm", {"prob": 0.1, "eta": 20.0}),
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


def get_operators_for_encoding(encoding: str) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    """Return available operators for the given encoding type."""
    return OPERATORS_BY_ENCODING.get(encoding, OPERATORS_BY_ENCODING["real"])


def get_crossover_names(encoding: str) -> List[str]:
    """Return list of crossover operator names for given encoding."""
    ops = get_operators_for_encoding(encoding)
    return [op[0] for op in ops.get("crossover", [])]


def get_mutation_names(encoding: str) -> List[str]:
    """Return list of mutation operator names for given encoding."""
    ops = get_operators_for_encoding(encoding)
    return [op[0] for op in ops.get("mutation", [])]


def normalize_operator_tuple(spec) -> tuple[str, dict] | None:
    """
    Accept operator specs in tuple/dict/string form and normalize to (name, params).
    """
    if spec is None:
        return None
    if isinstance(spec, tuple):
        return spec
    if isinstance(spec, str):
        return (spec, {})
    if isinstance(spec, dict):
        method = spec.get("method") or spec.get("name")
        if not method:
            return None
        params = {k: v for k, v in spec.items() if k not in {"method", "name"} and v is not None}
        return (method, params)
    return None


def normalize_variation_config(raw: dict | None) -> dict | None:
    """
    Normalize variation configuration dictionaries into operator tuples where applicable.
    """
    if not raw:
        return None
    normalized: dict = {}
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


def resolve_default_variation_config(encoding: str, overrides: dict | None) -> Dict[str, Any]:
    """
    Default variation configuration by encoding, merged with user overrides.
    """
    if encoding == "real":
        base = {
            "crossover": ("sbx", {"prob": 0.9, "eta": 20.0}),
            "mutation": ("pm", {"prob": 0.1, "eta": 20.0}),
        }
    elif encoding == "binary":
        base = {
            "crossover": ("hux", {"prob": 0.9}),
            "mutation": ("bitflip", {"prob": "1/n"}),
        }
    elif encoding == "integer":
        base = {
            "crossover": ("uniform", {"prob": 0.9}),
            "mutation": ("reset", {"prob": "1/n"}),
        }
    elif encoding == "mixed":
        base = {
            "crossover": ("mixed", {"prob": 0.9}),
            "mutation": ("mixed", {"prob": "1/n"}),
        }
    elif encoding == "permutation":
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


def merge_variation_overrides(base: dict | None, override: dict | None) -> dict:
    base = base or {}
    if not override:
        return dict(base)
    merged = dict(base)
    merged.update({k: v for k, v in override.items() if v is not None})
    return merged
