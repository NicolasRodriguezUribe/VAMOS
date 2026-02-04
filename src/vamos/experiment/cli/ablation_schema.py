"""
Ablation schema validation utilities.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


_ALLOWED_TOP_LEVEL_KEYS = {
    "version",
    "algorithm",
    "engine",
    "output_root",
    "default_max_evals",
    "max_evaluations",
    "problems",
    "seeds",
    "variants",
    "base_config",
    "per_variant_output_root",
    "output_root_by_variant",
    "budget_by_problem",
    "budget_by_variant",
    "budget_overrides",
    "metadata",
    "mirror_output_roots",
    "summary_dir",
    "summary_path",
}

_ALLOWED_VARIANT_KEYS = {
    "name",
    "label",
    "tags",
    "config_overrides",
    "nsgaii_variation",
    "moead_variation",
    "smsemoa_variation",
}

_OPERATOR_KEYS = {"crossover", "mutation", "selection", "repair", "aggregation"}
_BOOL_KEYS = {"steady_state", "use_numba_variation"}
_INT_KEYS = {"replacement_size", "k_neighbors", "archive_size", "n_partitions"}
_FLOAT_KEYS = {"kappa", "inertia", "c1", "c2", "vmax_fraction", "alpha", "adapt_freq"}
_STRING_KEYS = {"indicator"}
_MAPPING_KEYS = {"adaptive_operator_selection"}
_WEIGHT_KEYS = {"weight_vectors"}

_VARIATION_ALLOWED_KEYS = {
    "nsgaii": {"crossover", "mutation", "selection", "repair", "adaptive_operator_selection", "steady_state", "replacement_size"},
    "moead": {"crossover", "mutation", "aggregation", "repair", "weight_vectors", "use_numba_variation"},
    "smsemoa": {"crossover", "mutation", "selection", "repair"},
}


def _validate_mapping_values(value: Mapping[str, Any], name: str, value_type: type) -> None:
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings.")
        if not isinstance(item, value_type):
            raise TypeError(f"{name}[{key!r}] must be a {value_type.__name__}.")


def _validate_operator_spec(value: object, *, key: str) -> None:
    if value is None:
        return
    if isinstance(value, str):
        return
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Operator spec for '{key}' must be (name, params).")
        name, params = value
        if not isinstance(name, str):
            raise TypeError(f"Operator spec for '{key}' must start with a string name.")
        if not isinstance(params, Mapping):
            raise TypeError(f"Operator spec for '{key}' params must be a mapping.")
        return
    if isinstance(value, Mapping):
        if "method" not in value and "name" not in value:
            raise ValueError(f"Operator spec for '{key}' must include 'method' or 'name'.")
        return
    raise TypeError(f"Operator spec for '{key}' must be a string, tuple, or mapping.")


def _validate_variation_schema(variation: Mapping[str, Any], *, kind: str) -> None:
    allowed = _VARIATION_ALLOWED_KEYS[kind]
    unknown = set(variation) - allowed
    if unknown:
        raise ValueError(f"{kind}_variation has unsupported keys: {', '.join(sorted(unknown))}.")
    for key, value in variation.items():
        if value is None:
            continue
        if key in _OPERATOR_KEYS:
            _validate_operator_spec(value, key=key)
            continue
        if key in _MAPPING_KEYS:
            if not isinstance(value, Mapping):
                raise TypeError(f"{kind}_variation '{key}' must be a mapping.")
            continue
        if key in _WEIGHT_KEYS:
            if not isinstance(value, (Mapping, str)):
                raise TypeError(f"{kind}_variation '{key}' must be a mapping or string path.")
            continue
        if key in _BOOL_KEYS:
            if not isinstance(value, bool):
                raise TypeError(f"{kind}_variation '{key}' must be a boolean.")
            continue
        if key in _INT_KEYS:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{kind}_variation '{key}' must be an integer.")
            continue
        if key in _FLOAT_KEYS:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{kind}_variation '{key}' must be a number.")
            continue
        if key in _STRING_KEYS:
            if not isinstance(value, str):
                raise TypeError(f"{kind}_variation '{key}' must be a string.")
            continue


def validate_ablation_spec(spec: Mapping[str, Any]) -> None:
    unknown_top = set(spec) - _ALLOWED_TOP_LEVEL_KEYS
    if unknown_top:
        raise ValueError(f"Ablation config has unsupported keys: {', '.join(sorted(unknown_top))}.")

    required = ("problems", "variants", "seeds")
    for key in required:
        if key not in spec:
            raise ValueError(f"Ablation config missing required '{key}'.")

    problems = spec.get("problems")
    if not isinstance(problems, Iterable) or isinstance(problems, (str, bytes, Mapping)):
        raise ValueError("Ablation config 'problems' must be a non-empty list.")
    problems_list = list(problems)
    if not problems_list:
        raise ValueError("Ablation config 'problems' must be a non-empty list.")
    for item in problems_list:
        if not isinstance(item, str):
            raise TypeError("Ablation config 'problems' entries must be strings.")

    seeds = spec.get("seeds")
    if not isinstance(seeds, Iterable) or isinstance(seeds, (str, bytes, Mapping)):
        raise ValueError("Ablation config 'seeds' must be a non-empty list.")
    seeds_list = list(seeds)
    if not seeds_list:
        raise ValueError("Ablation config 'seeds' must be a non-empty list.")
    for item in seeds_list:
        if not isinstance(item, int):
            raise TypeError("Ablation config 'seeds' entries must be integers.")

    default_max_evals = spec.get("default_max_evals", spec.get("max_evaluations"))
    if not isinstance(default_max_evals, int) or default_max_evals <= 0:
        raise ValueError("Ablation config requires positive integer 'default_max_evals'.")

    for key in ("algorithm", "engine", "output_root", "summary_dir", "summary_path"):
        if key in spec and spec[key] is not None and not isinstance(spec[key], str):
            raise TypeError(f"Ablation config '{key}' must be a string when provided.")

    if "per_variant_output_root" in spec and not isinstance(spec["per_variant_output_root"], bool):
        raise TypeError("Ablation config 'per_variant_output_root' must be a boolean when provided.")

    base_config = spec.get("base_config")
    if base_config is not None and not isinstance(base_config, Mapping):
        raise TypeError("Ablation config 'base_config' must be a mapping when provided.")

    output_root_by_variant = spec.get("output_root_by_variant")
    if output_root_by_variant is not None:
        if not isinstance(output_root_by_variant, Mapping):
            raise TypeError("Ablation config 'output_root_by_variant' must be a mapping.")
        _validate_mapping_values(output_root_by_variant, "output_root_by_variant", str)

    budget_by_problem = spec.get("budget_by_problem")
    if budget_by_problem is not None:
        if not isinstance(budget_by_problem, Mapping):
            raise TypeError("Ablation config 'budget_by_problem' must be a mapping.")
        _validate_mapping_values(budget_by_problem, "budget_by_problem", int)

    budget_by_variant = spec.get("budget_by_variant")
    if budget_by_variant is not None:
        if not isinstance(budget_by_variant, Mapping):
            raise TypeError("Ablation config 'budget_by_variant' must be a mapping.")
        _validate_mapping_values(budget_by_variant, "budget_by_variant", int)

    budget_overrides = spec.get("budget_overrides")
    if budget_overrides is not None:
        if isinstance(budget_overrides, Mapping):
            for key, val in budget_overrides.items():
                if isinstance(val, Mapping):
                    _validate_mapping_values(val, f"budget_overrides[{key}]", int)
                else:
                    if not isinstance(val, int):
                        raise TypeError("budget_overrides values must be integers.")
        elif isinstance(budget_overrides, Iterable) and not isinstance(budget_overrides, (str, bytes)):
            for entry in budget_overrides:
                if not isinstance(entry, Mapping):
                    raise TypeError("budget_overrides list entries must be mappings.")
                if not isinstance(entry.get("problem"), str):
                    raise TypeError("budget_overrides entries require string 'problem'.")
                if not isinstance(entry.get("variant"), str):
                    raise TypeError("budget_overrides entries require string 'variant'.")
                if not isinstance(entry.get("max_evals"), int):
                    raise TypeError("budget_overrides entries require integer 'max_evals'.")
        else:
            raise TypeError("budget_overrides must be a mapping or list of mappings.")

    metadata = spec.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("Ablation config 'metadata' must be a mapping when provided.")

    mirror_output_roots = spec.get("mirror_output_roots")
    if mirror_output_roots is not None:
        if not isinstance(mirror_output_roots, Iterable) or isinstance(mirror_output_roots, (str, bytes, Mapping)):
            raise TypeError("mirror_output_roots must be a list of paths.")
        for item in mirror_output_roots:
            if not isinstance(item, str):
                raise TypeError("mirror_output_roots entries must be strings.")

    variants = spec.get("variants")
    if not isinstance(variants, Iterable) or isinstance(variants, (str, bytes, Mapping)):
        raise ValueError("Ablation config 'variants' must be a non-empty list.")
    variants_list = list(variants)
    if not variants_list:
        raise ValueError("Ablation config 'variants' must be a non-empty list.")
    for entry in variants_list:
        if not isinstance(entry, Mapping):
            raise TypeError("Each variant entry must be a mapping.")
        unknown_variant = set(entry) - _ALLOWED_VARIANT_KEYS
        if unknown_variant:
            raise ValueError(f"Variant has unsupported keys: {', '.join(sorted(unknown_variant))}.")
        if not isinstance(entry.get("name"), str) or not entry.get("name"):
            raise ValueError("Each variant entry requires non-empty string 'name'.")
        if "label" in entry and entry["label"] is not None and not isinstance(entry["label"], str):
            raise TypeError("Variant 'label' must be a string when provided.")
        if "tags" in entry:
            tags = entry["tags"]
            if isinstance(tags, str):
                raise TypeError("Variant 'tags' must be a list of strings, not a single string.")
            if not isinstance(tags, Iterable) or isinstance(tags, (bytes, Mapping)):
                raise TypeError("Variant 'tags' must be a list of strings.")
            for tag in tags:
                if not isinstance(tag, str):
                    raise TypeError("Variant 'tags' entries must be strings.")
        for key in ("config_overrides", "nsgaii_variation", "moead_variation", "smsemoa_variation"):
            if key in entry and entry[key] is not None and not isinstance(entry[key], Mapping):
                raise TypeError(f"Variant '{key}' must be a mapping when provided.")

        if "nsgaii_variation" in entry and entry["nsgaii_variation"] is not None:
            _validate_variation_schema(dict(entry["nsgaii_variation"]), kind="nsgaii")
        if "moead_variation" in entry and entry["moead_variation"] is not None:
            _validate_variation_schema(dict(entry["moead_variation"]), kind="moead")
        if "smsemoa_variation" in entry and entry["smsemoa_variation"] is not None:
            _validate_variation_schema(dict(entry["smsemoa_variation"]), kind="smsemoa")


__all__ = ["validate_ablation_spec"]
