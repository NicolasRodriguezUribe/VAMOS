"""
Parsing helpers for ablation configs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from vamos.engine.tuning.ablation import AblationVariant


def as_mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def as_sequence(value: object, name: str) -> list[Any]:
    if value is None:
        raise ValueError(f"{name} is required for ablation config.")
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a list.")
    return list(value)


def parse_budget_overrides(value: object) -> dict[tuple[str, str], int]:
    if value is None:
        return {}
    overrides: dict[tuple[str, str], int] = {}
    if isinstance(value, Mapping):
        for problem, inner in value.items():
            if isinstance(inner, Mapping):
                for variant, max_evals in inner.items():
                    overrides[(str(problem), str(variant))] = int(max_evals)
            else:
                key = str(problem)
                if ":" not in key:
                    raise ValueError("budget_overrides mapping keys must be 'problem:variant' or nested dicts.")
                prob, variant = key.split(":", 1)
                overrides[(prob.strip(), variant.strip())] = int(inner)
        return overrides
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for entry in value:
            if not isinstance(entry, Mapping):
                raise TypeError("budget_overrides list entries must be mappings.")
            problem = entry.get("problem")
            variant = entry.get("variant")
            max_evals = entry.get("max_evals")
            if problem is None or variant is None or max_evals is None:
                raise ValueError("budget_overrides entries require problem, variant, max_evals.")
            overrides[(str(problem), str(variant))] = int(max_evals)
        return overrides
    raise TypeError("budget_overrides must be a mapping or list of mappings.")


def normalize_variants(
    raw_variants: Sequence[Any],
    *,
    base_output_root: str | None,
    output_root_by_variant: Mapping[str, str],
    per_variant_output_root: bool,
) -> tuple[
    list[AblationVariant],
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
]:
    variants: list[AblationVariant] = []
    nsgaii_variations: dict[str, Mapping[str, Any]] = {}
    moead_variations: dict[str, Mapping[str, Any]] = {}
    smsemoa_variations: dict[str, Mapping[str, Any]] = {}
    for entry in raw_variants:
        if not isinstance(entry, Mapping):
            raise TypeError("Each variant entry must be a mapping.")
        name = entry.get("name")
        if not name:
            raise ValueError("Variant entry missing required 'name'.")
        label = entry.get("label")
        tags = entry.get("tags", ())
        if isinstance(tags, str):
            tags = (tags,)
        elif not isinstance(tags, Iterable):
            raise TypeError("Variant tags must be a list of strings.")

        overrides = as_mapping(entry.get("config_overrides"), "config_overrides")
        output_override = output_root_by_variant.get(str(name)) if output_root_by_variant else None
        if output_override:
            overrides.setdefault("output_root", str(output_override))
        elif base_output_root and per_variant_output_root:
            overrides.setdefault("output_root", str(Path(base_output_root) / str(name)))

        variant = AblationVariant(
            name=str(name),
            label=str(label) if label is not None else None,
            tags=tuple(str(tag) for tag in tags),
            config_overrides=overrides,
        )
        variants.append(variant)

        nsgaii_variation = entry.get("nsgaii_variation")
        if nsgaii_variation is not None:
            if not isinstance(nsgaii_variation, Mapping):
                raise TypeError("nsgaii_variation must be a mapping when provided.")
            nsgaii_variations[str(name)] = dict(nsgaii_variation)
        moead_variation = entry.get("moead_variation")
        if moead_variation is not None:
            if not isinstance(moead_variation, Mapping):
                raise TypeError("moead_variation must be a mapping when provided.")
            moead_variations[str(name)] = dict(moead_variation)
        smsemoa_variation = entry.get("smsemoa_variation")
        if smsemoa_variation is not None:
            if not isinstance(smsemoa_variation, Mapping):
                raise TypeError("smsemoa_variation must be a mapping when provided.")
            smsemoa_variations[str(name)] = dict(smsemoa_variation)

    return variants, nsgaii_variations, moead_variations, smsemoa_variations


__all__ = ["as_mapping", "as_sequence", "parse_budget_overrides", "normalize_variants"]
