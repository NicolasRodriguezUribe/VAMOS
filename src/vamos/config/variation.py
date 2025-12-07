"""
Shared helpers for variation/default handling across CLI, runner, and factories.
"""
from __future__ import annotations

from typing import Any, Dict


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


def resolve_nsgaii_variation_config(encoding: str, overrides: dict | None) -> Dict[str, Any]:
    """
    Default NSGA-II variation by encoding, merged with user overrides.
    """
    if encoding == "real":
        base = {
            "crossover": ("sbx", {"prob": 0.9, "eta": 20.0}),
            "mutation": ("pm", {"prob": 0.1, "eta": 20.0}),
        }
    elif encoding == "binary":
        base = {
            "crossover": ("hux", {"prob": 0.9}),
            "mutation": ("bitflip", {"prob": 0.1}),
        }
    elif encoding == "integer":
        base = {
            "crossover": ("sbx", {"prob": 0.9, "eta": 20.0}),
            "mutation": ("pm", {"prob": 0.1, "eta": 20.0}),
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
