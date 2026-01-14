"""
Experiment spec validation derived from the CLI parser + config dataclasses.

This keeps the YAML/JSON spec format in sync with the actual CLI flags and
algorithm config structures without maintaining a separate hard-coded whitelist.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from dataclasses import fields
from difflib import get_close_matches
from typing import Any, cast

from vamos.archive.bounded_archive import BoundedArchiveConfig
from vamos.engine.algorithm.config import (
    IBEAConfig,
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    SMPSOConfig,
    SMSEMOAConfig,
    SPEA2Config,
)
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import available_problem_names
from vamos.monitoring.hv_convergence import HVConvergenceConfig

from .spec_args import parser_spec_keys
from .spec_types import ExperimentSpec


EXPERIMENT_SPEC_VERSION = "1"

_ALGORITHM_BLOCKS: dict[str, type] = {
    "nsgaii": NSGAIIConfig,
    "moead": MOEADConfig,
    "smsemoa": SMSEMOAConfig,
    "nsgaiii": NSGAIIIConfig,
    "spea2": SPEA2Config,
    "ibea": IBEAConfig,
    "smpso": SMPSOConfig,
}

_OPERATOR_SPEC_KEYS = ("crossover", "mutation", "selection", "repair", "aggregation")


def validate_experiment_spec(spec: object, *, parser: argparse.ArgumentParser) -> ExperimentSpec:
    spec_dict = _as_str_dict(spec, path="Experiment spec")

    unknown = _unknown_keys(spec_dict, {"version", "defaults", "problems", "stopping", "archive"})
    if unknown:
        raise ValueError(f"Unknown top-level keys in experiment spec: {', '.join(unknown)}")

    version = spec_dict.get("version")
    if version is None:
        raise ValueError(f"Experiment spec must declare 'version: {EXPERIMENT_SPEC_VERSION}'.")
    version_str = str(version).strip()
    if version_str != EXPERIMENT_SPEC_VERSION:
        raise ValueError(f"Unsupported experiment spec version. Expected version={EXPERIMENT_SPEC_VERSION!r}, got {version_str!r}.")

    defaults = _as_str_dict(spec_dict.get("defaults"), path="Experiment spec 'defaults'")
    _validate_overrides_block(defaults, path="defaults", parser=parser)

    problems = _as_str_dict(spec_dict.get("problems"), path="Experiment spec 'problems'")
    known_problems = sorted(available_problem_names())
    for key, value in problems.items():
        normalized_key = key.lower()
        if normalized_key not in known_problems:
            suggestions = get_close_matches(normalized_key, known_problems, n=3, cutoff=0.6)
            suggestion_text = ""
            if suggestions:
                suggestion_text = f" Did you mean: {', '.join(suggestions)}?"
            raise ValueError(f"Unknown problem key in experiment spec: {key!r}.{suggestion_text}")
        if value is None:
            continue
        overrides = _as_str_dict(value, path=f"Experiment spec 'problems.{key}'")
        _validate_overrides_block(overrides, path=f"problems.{key}", parser=parser)

    if "stopping" in spec_dict:
        _validate_stopping_block(spec_dict.get("stopping"), path="stopping")
    if "archive" in spec_dict:
        _validate_archive_block(spec_dict.get("archive"), path="archive")

    return cast(ExperimentSpec, spec_dict)


def _validate_overrides_block(block: dict[str, Any], *, path: str, parser: argparse.ArgumentParser) -> None:
    allowed = _allowed_override_keys(parser)
    unknown = _unknown_keys(block, allowed)
    if unknown:
        raise ValueError(f"Unknown keys in '{path}': {', '.join(unknown)}")

    for algo_key, cfg_cls in _ALGORITHM_BLOCKS.items():
        if algo_key not in block:
            continue
        algo_block = block.get(algo_key)
        if algo_block is None:
            continue
        algo_dict = _as_str_dict(algo_block, path=f"'{path}.{algo_key}'")
        _validate_algorithm_block(algo_dict, config_cls=cfg_cls, path=f"{path}.{algo_key}")

    if "stopping" in block:
        _validate_stopping_block(block.get("stopping"), path=f"{path}.stopping")
    if "archive" in block:
        _validate_archive_block(block.get("archive"), path=f"{path}.archive")


def _allowed_override_keys(parser: argparse.ArgumentParser) -> set[str]:
    keys = set(parser_spec_keys(parser))
    keys.update(_dataclass_field_names(ExperimentConfig))
    keys.update(_ALGORITHM_BLOCKS.keys())
    keys.update({"stopping", "archive"})
    return keys


def _validate_algorithm_block(block: dict[str, Any], *, config_cls: type, path: str) -> None:
    allowed = set(_dataclass_field_names(config_cls))
    unknown = _unknown_keys(block, allowed)
    if unknown:
        raise ValueError(f"Unknown keys in '{path}': {', '.join(unknown)}")

    for op_key in _OPERATOR_SPEC_KEYS:
        if op_key not in block:
            continue
        spec = block.get(op_key)
        if spec is None:
            continue
        _validate_operator_spec(spec, path=f"{path}.{op_key}")

    aos = block.get("adaptive_operator_selection")
    if aos is not None and not isinstance(aos, Mapping):
        raise ValueError(f"'{path}.adaptive_operator_selection' must be a mapping when provided.")


def _validate_operator_spec(spec: object, *, path: str) -> None:
    if isinstance(spec, str):
        if not spec.strip():
            raise ValueError(f"'{path}' cannot be an empty operator name.")
        return
    if isinstance(spec, (tuple, list)):
        if len(spec) != 2:
            raise ValueError(f"'{path}' tuple/list form must have exactly 2 elements: [method, params].")
        method, params = spec
        if not isinstance(method, str) or not method.strip():
            raise ValueError(f"'{path}' operator method must be a non-empty string.")
        if not isinstance(params, Mapping):
            raise ValueError(f"'{path}' operator params must be a mapping.")
        return
    if isinstance(spec, Mapping):
        spec_map = _as_str_dict(spec, path=f"'{path}'")
        method = spec_map.get("method") or spec_map.get("name")
        if not isinstance(method, str) or not method.strip():
            raise ValueError(f"'{path}' mapping form requires a non-empty 'method' (or 'name') key.")
        return
    raise ValueError(f"'{path}' must be a string, a 2-tuple/list, or a mapping with 'method'.")


def _validate_stopping_block(block: object, *, path: str) -> None:
    if block is None:
        return
    block_dict = _as_str_dict(block, path=f"'{path}'")
    unknown = _unknown_keys(block_dict, {"hv_convergence"})
    if unknown:
        raise ValueError(f"Unknown keys in '{path}': {', '.join(unknown)}")
    hv = block_dict.get("hv_convergence")
    if hv is None:
        return
    hv_dict = _as_str_dict(hv, path=f"'{path}.hv_convergence'")
    allowed = {"enabled", "ref_point", *_dataclass_field_names(HVConvergenceConfig)}
    unknown_hv = _unknown_keys(hv_dict, allowed)
    if unknown_hv:
        raise ValueError(f"Unknown keys in '{path}.hv_convergence': {', '.join(unknown_hv)}")

    ref = hv_dict.get("ref_point")
    if ref is not None and not (
        (isinstance(ref, str) and ref.lower() == "auto") or (isinstance(ref, (list, tuple)) and all(_is_number(x) for x in ref))
    ):
        raise ValueError(f"'{path}.hv_convergence.ref_point' must be 'auto' or a list of numbers.")


def _validate_archive_block(block: object, *, path: str) -> None:
    if block is None:
        return
    block_dict = _as_str_dict(block, path=f"'{path}'")
    unknown = _unknown_keys(block_dict, {"bounded"})
    if unknown:
        raise ValueError(f"Unknown keys in '{path}': {', '.join(unknown)}")
    bounded = block_dict.get("bounded")
    if bounded is None:
        return
    bounded_dict = _as_str_dict(bounded, path=f"'{path}.bounded'")
    allowed = set(_dataclass_field_names(BoundedArchiveConfig))
    unknown_bounded = _unknown_keys(bounded_dict, allowed)
    if unknown_bounded:
        raise ValueError(f"Unknown keys in '{path}.bounded': {', '.join(unknown_bounded)}")


def _dataclass_field_names(cls: object) -> list[str]:
    return [field.name for field in fields(cls)]  # type: ignore[arg-type]


def _unknown_keys(d: Mapping[str, Any], allowed: Iterable[str]) -> list[str]:
    allowed_set = set(allowed)
    return sorted(key for key in d.keys() if key not in allowed_set)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_str_dict(value: object, *, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping (YAML/JSON object).")
    out: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{path} keys must be non-empty strings.")
        out[key] = item
    return out
