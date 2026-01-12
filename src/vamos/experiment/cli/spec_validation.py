"""
Experiment spec validation derived from the CLI parser + config dataclasses.

This keeps the YAML/JSON spec format in sync with the actual CLI flags and
algorithm config structures without maintaining a separate hard-coded whitelist.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from typing import Any, Iterable

from vamos.archive.bounded_archive import BoundedArchiveConfig
from vamos.engine.algorithm.config import (
    IBEAConfigData,
    MOEADConfigData,
    NSGAIIConfigData,
    NSGAIIIConfigData,
    SMPSOConfigData,
    SMSEMOAConfigData,
    SPEA2ConfigData,
)
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.monitoring.hv_convergence import HVConvergenceConfig

from .spec_args import parser_spec_keys


EXPERIMENT_SPEC_VERSION = "1"

_ALGORITHM_BLOCKS: dict[str, type] = {
    "nsgaii": NSGAIIConfigData,
    "moead": MOEADConfigData,
    "smsemoa": SMSEMOAConfigData,
    "nsgaiii": NSGAIIIConfigData,
    "spea2": SPEA2ConfigData,
    "ibea": IBEAConfigData,
    "smpso": SMPSOConfigData,
}

_OPERATOR_SPEC_KEYS = ("crossover", "mutation", "selection", "repair", "aggregation")


def validate_experiment_spec(spec: object, *, parser: argparse.ArgumentParser) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise TypeError("Experiment spec must be a mapping (YAML/JSON object).")

    unknown = _unknown_keys(spec, {"version", "defaults", "problems", "stopping", "archive"})
    if unknown:
        raise ValueError(f"Unknown top-level keys in experiment spec: {', '.join(unknown)}")

    version = spec.get("version")
    if version is not None:
        version_str = str(version).strip()
        if version_str != EXPERIMENT_SPEC_VERSION:
            raise ValueError(f"Unsupported experiment spec version. Expected version={EXPERIMENT_SPEC_VERSION!r}, got {version_str!r}.")

    defaults = spec.get("defaults")
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise ValueError("Experiment spec 'defaults' must be a mapping when provided.")
    _validate_overrides_block(defaults, path="defaults", parser=parser)

    problems = spec.get("problems")
    if problems is None:
        problems = {}
    if not isinstance(problems, dict):
        raise ValueError("Experiment spec 'problems' must be a mapping of problem_key -> overrides.")
    for key, value in problems.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Experiment spec 'problems' keys must be non-empty strings.")
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"Experiment spec 'problems.{key}' must be a mapping of overrides.")
        _validate_overrides_block(value, path=f"problems.{key}", parser=parser)

    if "stopping" in spec:
        _validate_stopping_block(spec.get("stopping"), path="stopping")
    if "archive" in spec:
        _validate_archive_block(spec.get("archive"), path="archive")

    return spec


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
        if not isinstance(algo_block, dict):
            raise ValueError(f"'{path}.{algo_key}' must be a mapping when provided.")
        _validate_algorithm_block(algo_block, config_cls=cfg_cls, path=f"{path}.{algo_key}")

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
    if aos is not None and not isinstance(aos, dict):
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
        if not isinstance(params, dict):
            raise ValueError(f"'{path}' operator params must be a mapping.")
        return
    if isinstance(spec, dict):
        method = spec.get("method") or spec.get("name")
        if not isinstance(method, str) or not method.strip():
            raise ValueError(f"'{path}' mapping form requires a non-empty 'method' (or 'name') key.")
        return
    raise ValueError(f"'{path}' must be a string, a 2-tuple/list, or a mapping with 'method'.")


def _validate_stopping_block(block: object, *, path: str) -> None:
    if block is None:
        return
    if not isinstance(block, dict):
        raise ValueError(f"'{path}' must be a mapping when provided.")
    unknown = _unknown_keys(block, {"hv_convergence"})
    if unknown:
        raise ValueError(f"Unknown keys in '{path}': {', '.join(unknown)}")
    hv = block.get("hv_convergence")
    if hv is None:
        return
    if not isinstance(hv, dict):
        raise ValueError(f"'{path}.hv_convergence' must be a mapping when provided.")
    allowed = {"enabled", "ref_point", *_dataclass_field_names(HVConvergenceConfig)}
    unknown_hv = _unknown_keys(hv, allowed)
    if unknown_hv:
        raise ValueError(f"Unknown keys in '{path}.hv_convergence': {', '.join(unknown_hv)}")

    ref = hv.get("ref_point")
    if ref is not None and not (
        (isinstance(ref, str) and ref.lower() == "auto") or (isinstance(ref, (list, tuple)) and all(_is_number(x) for x in ref))
    ):
        raise ValueError(f"'{path}.hv_convergence.ref_point' must be 'auto' or a list of numbers.")


def _validate_archive_block(block: object, *, path: str) -> None:
    if block is None:
        return
    if not isinstance(block, dict):
        raise ValueError(f"'{path}' must be a mapping when provided.")
    unknown = _unknown_keys(block, {"bounded"})
    if unknown:
        raise ValueError(f"Unknown keys in '{path}': {', '.join(unknown)}")
    bounded = block.get("bounded")
    if bounded is None:
        return
    if not isinstance(bounded, dict):
        raise ValueError(f"'{path}.bounded' must be a mapping when provided.")
    allowed = set(_dataclass_field_names(BoundedArchiveConfig))
    unknown_bounded = _unknown_keys(bounded, allowed)
    if unknown_bounded:
        raise ValueError(f"Unknown keys in '{path}.bounded': {', '.join(unknown_bounded)}")


def _dataclass_field_names(cls: object) -> list[str]:
    return [field.name for field in fields(cls)]  # type: ignore[arg-type]


def _unknown_keys(d: dict[str, Any], allowed: Iterable[str]) -> list[str]:
    allowed_set = set(allowed)
    return sorted(key for key in d.keys() if key not in allowed_set)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
