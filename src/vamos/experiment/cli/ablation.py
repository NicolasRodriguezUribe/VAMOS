from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from vamos.engine.config.loader import load_experiment_spec
from vamos.engine.tuning.ablation import AblationVariant, build_ablation_plan
from vamos.experiment.study.api import run_ablation_plan
from vamos.foundation.core.experiment_config import ExperimentConfig

from .ablation_parse import as_mapping, as_sequence, normalize_variants, parse_budget_overrides
from .ablation_schema import validate_ablation_spec
from .ablation_summary import write_summary_csv


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


def _coerce_output_root_by_variant(value: Mapping[str, Any]) -> dict[str, str]:
    return {str(k): str(v) for k, v in value.items()}


def _coerce_budget_mapping(value: Mapping[str, Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in value.items()}


def _normalize_variant_payloads(
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
    return normalize_variants(
        raw_variants,
        base_output_root=base_output_root,
        output_root_by_variant=output_root_by_variant,
        per_variant_output_root=per_variant_output_root,
    )


def run_ablation(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    raw = load_experiment_spec(args.config)
    if not isinstance(raw, Mapping):
        raise TypeError("Ablation config must be a mapping (YAML/JSON object).")
    validate_ablation_spec(raw)

    algorithm = str(raw.get("algorithm", "nsgaii")).lower()
    engine = raw.get("engine")
    base_config = as_mapping(raw.get("base_config"), "base_config")

    base_output_root = args.output_root or raw.get("output_root") or base_config.get("output_root")
    per_variant_output_root = bool(raw.get("per_variant_output_root", True))
    if args.flat_output:
        per_variant_output_root = False
    if per_variant_output_root and not base_output_root:
        base_output_root = ExperimentConfig().output_root

    output_root_by_variant = as_mapping(raw.get("output_root_by_variant"), "output_root_by_variant")
    output_root_by_variant = _coerce_output_root_by_variant(output_root_by_variant)
    if per_variant_output_root and "output_root" in base_config:
        base_config.pop("output_root", None)
    elif base_output_root:
        base_config["output_root"] = str(base_output_root)

    problems = [str(p) for p in as_sequence(raw.get("problems"), "problems")]
    seeds = [int(s) for s in as_sequence(raw.get("seeds"), "seeds")]
    default_max_evals = int(raw.get("default_max_evals") or raw.get("max_evaluations") or 0)
    if default_max_evals <= 0:
        raise ValueError("default_max_evals must be a positive integer.")

    raw_variants = as_sequence(raw.get("variants"), "variants")
    variants, nsgaii_variations, moead_variations, smsemoa_variations = _normalize_variant_payloads(
        raw_variants,
        base_output_root=str(base_output_root) if base_output_root else None,
        output_root_by_variant=output_root_by_variant,
        per_variant_output_root=per_variant_output_root,
    )

    budget_by_problem = _coerce_budget_mapping(as_mapping(raw.get("budget_by_problem"), "budget_by_problem"))
    budget_by_variant = _coerce_budget_mapping(as_mapping(raw.get("budget_by_variant"), "budget_by_variant"))
    budget_overrides = parse_budget_overrides(raw.get("budget_overrides"))
    metadata = as_mapping(raw.get("metadata"), "metadata")

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
        write_summary_csv(results, variant_names, summary_path_obj)
        _logger().info("[Ablation] Summary CSV: %s", summary_path_obj)

    _logger().info("[Ablation] Completed %s tasks.", plan.n_tasks)


__all__ = ["run_ablation"]
