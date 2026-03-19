from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vamos.experiment.benchmark.report_utils import ensure_dir
from vamos.experiment.study.types import StudyResult

ARCHIVE_FAMILY_SUITE_PREFIX = "NSGAII_archive_family"
ARCHIVE_FAMILY_DEFAULT_ALGORITHMS = [
    "nsgaii_archive_off",
    "nsgaii_archive_passive",
    "nsgaii_archive_hybrid",
]
ARCHIVE_FAMILY_DEFAULT_METRICS = [
    "hv",
    "igd_plus",
    "archive_subset_hv",
    "archive_subset_igd_plus",
]


@dataclass(frozen=True)
class BenchmarkAlgorithmAlias:
    name: str
    execution_algorithm: str
    label: str
    output_root_suffix: str
    nsgaii_variation: dict[str, Any] | None = None


_ARCHIVE_FAMILY_ALIASES: dict[str, BenchmarkAlgorithmAlias] = {
    "nsgaii_archive_off": BenchmarkAlgorithmAlias(
        name="nsgaii_archive_off",
        execution_algorithm="nsgaii",
        label="nsgaii_archive_off",
        output_root_suffix="nsgaii_archive_off",
        nsgaii_variation={"archive_mode": "off"},
    ),
    "nsgaii_archive_passive": BenchmarkAlgorithmAlias(
        name="nsgaii_archive_passive",
        execution_algorithm="nsgaii",
        label="nsgaii_archive_passive",
        output_root_suffix="nsgaii_archive_passive",
        nsgaii_variation={"archive_mode": "passive"},
    ),
    "nsgaii_archive_hybrid": BenchmarkAlgorithmAlias(
        name="nsgaii_archive_hybrid",
        execution_algorithm="nsgaii",
        label="nsgaii_archive_hybrid",
        output_root_suffix="nsgaii_archive_hybrid",
        nsgaii_variation={"archive_mode": "hybrid_survival"},
    ),
}


def resolve_benchmark_algorithm_alias(name: str) -> BenchmarkAlgorithmAlias | None:
    return _ARCHIVE_FAMILY_ALIASES.get(str(name))


def is_archive_family_suite(name: str) -> bool:
    return str(name).startswith(ARCHIVE_FAMILY_SUITE_PREFIX)


def _selected_archive_family_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem": row.get("problem"),
        "problem_label": row.get("problem_label"),
        "n_var": row.get("n_var"),
        "n_obj": row.get("n_obj"),
        "algorithm": row.get("algorithm"),
        "algorithm_base": row.get("algorithm_base"),
        "engine": row.get("engine"),
        "seed": row.get("seed"),
        "evaluations": row.get("evaluations"),
        "time_ms": row.get("time_ms"),
        "hv": row.get("hv"),
        "indicator_igd_plus": row.get("indicator_igd_plus"),
        "archive_subset_hv": row.get("archive_subset_hv"),
        "archive_subset_igd_plus": row.get("archive_subset_igd_plus"),
        "archive_mode": row.get("archive_mode"),
        "archive_execution_mode": row.get("archive_execution_mode"),
        "archive_survival_path": row.get("archive_survival_path"),
        "archive_size": row.get("archive_size"),
        "archive_subset_size": row.get("archive_subset_size"),
        "hybrid_status": row.get("hybrid_status"),
        "hybrid_fallback_reason": row.get("hybrid_fallback_reason"),
        "hybrid_split_front_mode": row.get("hybrid_split_front_mode"),
        "hybrid_split_front_reason": row.get("hybrid_split_front_reason"),
        "hybrid_generations": row.get("hybrid_generations"),
        "hybrid_archive_reference_generations": row.get("hybrid_archive_reference_generations"),
        "hybrid_local_only_generations": row.get("hybrid_local_only_generations"),
        "hybrid_no_split_generations": row.get("hybrid_no_split_generations"),
        "output_dir": row.get("output_dir"),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    if not rows:
        return path
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _mean(total: float, count: int) -> float | None:
    return (total / count) if count > 0 else None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_archive_family_summary(results: list[StudyResult], output_dir: Path) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    for result in results:
        row = result.to_row()
        if row.get("algorithm_base") != "nsgaii":
            continue
        rows.append(_selected_archive_family_row(row))

    if not rows:
        return {}

    runs_path = _write_csv(rows, output_dir / "archive_family_runs.csv")

    grouped: dict[tuple[str, Any, str], dict[str, Any]] = defaultdict(
        lambda: {
            "runs": 0,
            "hv_total": 0.0,
            "hv_count": 0,
            "igd_plus_total": 0.0,
            "igd_plus_count": 0,
            "archive_subset_hv_total": 0.0,
            "archive_subset_hv_count": 0,
            "archive_subset_igd_plus_total": 0.0,
            "archive_subset_igd_plus_count": 0,
            "time_ms_total": 0.0,
            "archive_size_total": 0.0,
            "archive_subset_size_total": 0.0,
            "hybrid_archive_reference_generations_total": 0.0,
            "hybrid_local_only_generations_total": 0.0,
            "hybrid_no_split_generations_total": 0.0,
            "hybrid_active_runs": 0,
            "hybrid_fallback_runs": 0,
            "hybrid_local_only_runs": 0,
        }
    )

    by_variant: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "runs": 0,
            "hybrid_active_runs": 0,
            "hybrid_fallback_runs": 0,
            "hybrid_local_only_runs": 0,
            "archive_reference_generations_total": 0,
            "local_only_generations_total": 0,
            "no_split_generations_total": 0,
        }
    )

    for row in rows:
        key = (str(row.get("problem")), row.get("n_obj"), str(row.get("algorithm")))
        agg = grouped[key]
        agg["runs"] += 1
        for metric_name in ("hv", "indicator_igd_plus", "archive_subset_hv", "archive_subset_igd_plus"):
            value = _to_float(row.get(metric_name))
            if value is not None:
                total_key = {
                    "hv": "hv_total",
                    "indicator_igd_plus": "igd_plus_total",
                    "archive_subset_hv": "archive_subset_hv_total",
                    "archive_subset_igd_plus": "archive_subset_igd_plus_total",
                }[metric_name]
                count_key = {
                    "hv": "hv_count",
                    "indicator_igd_plus": "igd_plus_count",
                    "archive_subset_hv": "archive_subset_hv_count",
                    "archive_subset_igd_plus": "archive_subset_igd_plus_count",
                }[metric_name]
                agg[total_key] += value
                agg[count_key] += 1
        for metric_name, target_key in (
            ("time_ms", "time_ms_total"),
            ("archive_size", "archive_size_total"),
            ("archive_subset_size", "archive_subset_size_total"),
            ("hybrid_archive_reference_generations", "hybrid_archive_reference_generations_total"),
            ("hybrid_local_only_generations", "hybrid_local_only_generations_total"),
            ("hybrid_no_split_generations", "hybrid_no_split_generations_total"),
        ):
            value = _to_float(row.get(metric_name))
            if value is not None:
                agg[target_key] += value

        variant = by_variant[str(row.get("algorithm"))]
        variant["runs"] += 1
        if row.get("archive_survival_path") == "hybrid":
            variant["hybrid_active_runs"] += 1
        if row.get("hybrid_fallback_reason"):
            agg["hybrid_fallback_runs"] += 1
            variant["hybrid_fallback_runs"] += 1
        if row.get("hybrid_split_front_mode") == "local_only":
            agg["hybrid_local_only_runs"] += 1
            variant["hybrid_local_only_runs"] += 1
        variant["archive_reference_generations_total"] += int(row.get("hybrid_archive_reference_generations") or 0)
        variant["local_only_generations_total"] += int(row.get("hybrid_local_only_generations") or 0)
        variant["no_split_generations_total"] += int(row.get("hybrid_no_split_generations") or 0)

    means_rows: list[dict[str, Any]] = []
    for (problem, n_obj, algorithm), agg in sorted(grouped.items()):
        means_rows.append(
            {
                "problem": problem,
                "n_obj": n_obj,
                "algorithm": algorithm,
                "runs": agg["runs"],
                "mean_hv": _mean(agg["hv_total"], agg["hv_count"]),
                "mean_igd_plus": _mean(agg["igd_plus_total"], agg["igd_plus_count"]),
                "mean_archive_subset_hv": _mean(agg["archive_subset_hv_total"], agg["archive_subset_hv_count"]),
                "mean_archive_subset_igd_plus": _mean(
                    agg["archive_subset_igd_plus_total"], agg["archive_subset_igd_plus_count"]
                ),
                "mean_time_ms": _mean(agg["time_ms_total"], agg["runs"]),
                "mean_archive_size": _mean(agg["archive_size_total"], agg["runs"]),
                "mean_archive_subset_size": _mean(agg["archive_subset_size_total"], agg["runs"]),
                "mean_hybrid_archive_reference_generations": _mean(
                    agg["hybrid_archive_reference_generations_total"], agg["runs"]
                ),
                "mean_hybrid_local_only_generations": _mean(agg["hybrid_local_only_generations_total"], agg["runs"]),
                "mean_hybrid_no_split_generations": _mean(agg["hybrid_no_split_generations_total"], agg["runs"]),
                "hybrid_fallback_runs": agg["hybrid_fallback_runs"],
                "hybrid_local_only_runs": agg["hybrid_local_only_runs"],
            }
        )

    means_path = _write_csv(means_rows, output_dir / "archive_family_means.csv")

    summary_payload = {
        "runs": len(rows),
        "variants": sorted(by_variant.keys()),
        "by_variant": dict(sorted(by_variant.items())),
        "by_problem_objectives_variant": means_rows,
    }
    summary_path = output_dir / "archive_family_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return {
        "runs": runs_path,
        "means": means_path,
        "summary": summary_path,
    }


__all__ = [
    "ARCHIVE_FAMILY_DEFAULT_ALGORITHMS",
    "ARCHIVE_FAMILY_DEFAULT_METRICS",
    "ARCHIVE_FAMILY_SUITE_PREFIX",
    "BenchmarkAlgorithmAlias",
    "is_archive_family_suite",
    "resolve_benchmark_algorithm_alias",
    "write_archive_family_summary",
]
