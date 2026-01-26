from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping, Sequence

from vamos.engine.config.loader import load_experiment_spec
from vamos.engine.tuning.ablation import AblationVariant, build_ablation_plan
from vamos.experiment.study.api import run_ablation_plan
from vamos.foundation.core.experiment_config import ExperimentConfig


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ablation plans from a YAML/JSON config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to ablation config (YAML/JSON).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override output root for all variants (base path).",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Do not create per-variant output roots (advanced; may overwrite outputs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan summary and exit without running experiments.",
    )
    return parser


def _as_mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def _as_sequence(value: object, name: str) -> list[Any]:
    if value is None:
        raise ValueError(f"{name} is required for ablation config.")
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a list.")
    return list(value)


def _validate_mapping_values(value: Mapping[str, Any], name: str, value_type: type) -> None:
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings.")
        if not isinstance(item, value_type):
            raise TypeError(f"{name}[{key!r}] must be a {value_type.__name__}.")


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


def _validate_ablation_spec(spec: Mapping[str, Any]) -> None:
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


def _parse_budget_overrides(value: object) -> dict[tuple[str, str], int]:
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


def _normalize_variants(
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

        overrides = _as_mapping(entry.get("config_overrides"), "config_overrides")
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


def _write_summary_csv(
    results: Sequence[Any],
    variant_names: Sequence[str],
    path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for result, variant in zip(results, variant_names):
        row = result.to_row()
        row["variant"] = variant
        rows.append(row)

    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_ablation(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    raw = load_experiment_spec(args.config)
    if not isinstance(raw, Mapping):
        raise TypeError("Ablation config must be a mapping (YAML/JSON object).")
    _validate_ablation_spec(raw)

    algorithm = str(raw.get("algorithm", "nsgaii")).lower()
    engine = raw.get("engine")
    base_config = _as_mapping(raw.get("base_config"), "base_config")

    base_output_root = args.output_root or raw.get("output_root") or base_config.get("output_root")
    per_variant_output_root = bool(raw.get("per_variant_output_root", True))
    if args.flat_output:
        per_variant_output_root = False
    if per_variant_output_root and not base_output_root:
        base_output_root = ExperimentConfig().output_root

    output_root_by_variant = _as_mapping(raw.get("output_root_by_variant"), "output_root_by_variant")
    if per_variant_output_root and "output_root" in base_config:
        base_config.pop("output_root", None)
    elif base_output_root:
        base_config["output_root"] = str(base_output_root)

    problems = [str(p) for p in _as_sequence(raw.get("problems"), "problems")]
    seeds = [int(s) for s in _as_sequence(raw.get("seeds"), "seeds")]
    default_max_evals = int(raw.get("default_max_evals") or raw.get("max_evaluations") or 0)
    if default_max_evals <= 0:
        raise ValueError("default_max_evals must be a positive integer.")

    raw_variants = _as_sequence(raw.get("variants"), "variants")
    variants, nsgaii_variations, moead_variations, smsemoa_variations = _normalize_variants(
        raw_variants,
        base_output_root=str(base_output_root) if base_output_root else None,
        output_root_by_variant=output_root_by_variant,
        per_variant_output_root=per_variant_output_root,
    )

    budget_by_problem = {str(k): int(v) for k, v in _as_mapping(raw.get("budget_by_problem"), "budget_by_problem").items()}
    budget_by_variant = {str(k): int(v) for k, v in _as_mapping(raw.get("budget_by_variant"), "budget_by_variant").items()}
    budget_overrides = _parse_budget_overrides(raw.get("budget_overrides"))
    metadata = _as_mapping(raw.get("metadata"), "metadata")

    plan = build_ablation_plan(
        problems=problems,
        variants=variants,
        seeds=seeds,
        default_max_evals=default_max_evals,
        engine=str(engine) if engine is not None else None,
        budget_by_problem=budget_by_problem or None,
        budget_by_variant=budget_by_variant or None,
        budget_overrides=budget_overrides or None,
        metadata=metadata or None,
    )

    if args.dry_run:
        _logger().info("[Ablation] %s tasks (%s variants, %s problems, %s seeds).", plan.n_tasks, len(variants), len(problems), len(seeds))
        for task in plan.tasks[: min(5, plan.n_tasks)]:
            _logger().info("[Ablation] Task: %s", task.as_dict())
        return

    mirror_output_roots = raw.get("mirror_output_roots")
    mirror = None
    if mirror_output_roots is not None:
        if not isinstance(mirror_output_roots, Iterable) or isinstance(mirror_output_roots, (str, bytes, Mapping)):
            raise TypeError("mirror_output_roots must be a list of paths.")
        mirror = tuple(str(p) for p in mirror_output_roots)

    if algorithm != "nsgaii" and nsgaii_variations:
        _logger().warning("[Ablation] nsgaii_variation provided but algorithm=%s; ignoring variations.", algorithm)
        nsgaii_variations = {}
    if algorithm != "moead" and moead_variations:
        _logger().warning("[Ablation] moead_variation provided but algorithm=%s; ignoring variations.", algorithm)
        moead_variations = {}
    if algorithm != "smsemoa" and smsemoa_variations:
        _logger().warning("[Ablation] smsemoa_variation provided but algorithm=%s; ignoring variations.", algorithm)
        smsemoa_variations = {}

    results, variant_names = run_ablation_plan(
        plan,
        algorithm=algorithm,
        base_config=base_config,
        nsgaii_variations=nsgaii_variations or None,
        moead_variations=moead_variations or None,
        smsemoa_variations=smsemoa_variations or None,
        engine=str(engine) if engine is not None else None,
        mirror_output_roots=mirror,
    )

    summary_path = raw.get("summary_path")
    summary_dir = raw.get("summary_dir")
    summary_path_obj: Path | None
    if summary_path:
        summary_path_obj = Path(str(summary_path))
    elif summary_dir:
        summary_path_obj = Path(str(summary_dir)) / "ablation_metrics.csv"
    else:
        summary_root = Path(str(base_output_root or ExperimentConfig().output_root))
        summary_path_obj = summary_root / "summary" / "ablation_metrics.csv"

    if summary_path_obj is not None:
        _write_summary_csv(results, variant_names, summary_path_obj)
        _logger().info("[Ablation] Summary CSV: %s", summary_path_obj)

    _logger().info("[Ablation] Completed %s tasks.", plan.n_tasks)


__all__ = ["run_ablation"]
