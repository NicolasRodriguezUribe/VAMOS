from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from vamos.experiment._online_control_analysis_report import (
    build_analysis_report,
    build_go_no_go_analysis,
    build_heterogeneity_summary,
)

FAMILY_LABELS = ("sbx_like", "de_like")
REGIME_LABELS = ("repair", "expand", "refine")
INTENT_LABELS = ("exploratory", "balanced", "local_refine", "mutation_heavy", "feasibility_biased")
ADAPTIVE_VARIANTS = ("adaptive_flat_operator", "adaptive_flat_parameter", "adaptive_hierarchical_joint")
BASELINE_BY_HOST = {"nsgaii": "fixed_sbx", "moead": "fixed_de"}


def _float_or_default(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_scalar(value: str) -> Any:
    raw = value.strip()
    if raw == "":
        return None
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if raw.startswith("{") or raw.startswith("["):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    try:
        if any(token in raw for token in (".", "e", "E")):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [{key: _coerce_scalar(value) for key, value in row.items()} for row in reader]


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: value if value is None or isinstance(value, (str, int, float, bool)) else json.dumps(value, sort_keys=True)
                    for key, value in row.items()
                }
            )


def load_pilot_output(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        read_csv_rows(output_dir / "runs.csv"),
        read_csv_rows(output_dir / "summary.csv"),
        read_csv_rows(output_dir / "trace_rows.csv"),
    )


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def phase_from_progress(budget_progress: float | None) -> str:
    value = 0.0 if budget_progress is None else float(budget_progress)
    if value < (1.0 / 3.0):
        return "early"
    if value < (2.0 / 3.0):
        return "mid"
    return "late"


def _entropy_from_shares(shares: list[float]) -> float:
    positive = [share for share in shares if share > 0.0]
    if not positive:
        return 0.0
    return float(-sum(share * math.log(share) for share in positive))


def _best_fixed_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    fixed = [row for row in rows if str(row.get("variant", "")).startswith("fixed_")]
    if not fixed:
        return None
    return max(
        fixed,
        key=lambda row: (
            _float_or_default(row.get("mean_hv")),
            -(_float_or_default(row.get("mean_igd_plus"))) if row.get("mean_igd_plus") is not None else float("-inf"),
            -_float_or_default(row.get("mean_time_ms")),
        ),
    )


def compute_problem_host_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(str(row.get("host")), str(row.get("problem")))].append(dict(row))

    output: list[dict[str, Any]] = []
    hv_tolerance = 1e-6
    for (host, problem), rows in sorted(grouped.items()):
        best_fixed = _best_fixed_row(rows)
        for row in rows:
            enriched = dict(row)
            if best_fixed is None:
                enriched["best_fixed_variant"] = None
                enriched["hv_gap_vs_best_fixed"] = None
                enriched["igd_plus_gap_vs_best_fixed"] = None
                enriched["runtime_ratio_vs_best_fixed"] = None
                enriched["comparison_to_best_fixed"] = "no_best_fixed"
            else:
                enriched["best_fixed_variant"] = best_fixed.get("variant")
                hv_gap = float(row.get("mean_hv") or 0.0) - float(best_fixed.get("mean_hv") or 0.0)
                enriched["hv_gap_vs_best_fixed"] = hv_gap
                if row.get("mean_igd_plus") is not None and best_fixed.get("mean_igd_plus") is not None:
                    enriched["igd_plus_gap_vs_best_fixed"] = float(row["mean_igd_plus"]) - float(best_fixed["mean_igd_plus"])
                else:
                    enriched["igd_plus_gap_vs_best_fixed"] = None
                baseline_time = max(1e-9, float(best_fixed.get("mean_time_ms") or 0.0))
                enriched["runtime_ratio_vs_best_fixed"] = float(row.get("mean_time_ms") or 0.0) / baseline_time
                if row.get("variant") == best_fixed.get("variant"):
                    comparison = "best_fixed"
                elif hv_gap > hv_tolerance:
                    comparison = "win"
                elif hv_gap < -hv_tolerance:
                    comparison = "loss"
                else:
                    comparison = "tie"
                enriched["comparison_to_best_fixed"] = comparison
            output.append(enriched)
    return output


def build_policy_comparison(problem_host_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in problem_host_rows:
        if row.get("variant") in ADAPTIVE_VARIANTS:
            grouped[(str(row["host"]), str(row["problem"]))][str(row["variant"])] = row

    comparisons: list[dict[str, Any]] = []
    hv_tolerance = 1e-6
    for (host, problem), mapping in sorted(grouped.items()):
        variants = [name for name in ADAPTIVE_VARIANTS if name in mapping]
        for left in variants:
            for right in variants:
                if left == right:
                    continue
                left_row = mapping[left]
                right_row = mapping[right]
                hv_gap = float(left_row.get("mean_hv") or 0.0) - float(right_row.get("mean_hv") or 0.0)
                if hv_gap > hv_tolerance:
                    outcome = "win"
                elif hv_gap < -hv_tolerance:
                    outcome = "loss"
                else:
                    outcome = "tie"
                comparisons.append(
                    {
                        "host": host,
                        "problem": problem,
                        "left_variant": left,
                        "right_variant": right,
                        "hv_gap": hv_gap,
                        "igd_plus_gap": (
                            float(left_row["mean_igd_plus"]) - float(right_row["mean_igd_plus"])
                            if left_row.get("mean_igd_plus") is not None and right_row.get("mean_igd_plus") is not None
                            else None
                        ),
                        "runtime_ratio": float(left_row.get("mean_time_ms") or 0.0) / max(1e-9, float(right_row.get("mean_time_ms") or 0.0)),
                        "reward_gap": (
                            float(left_row["mean_average_reward"]) - float(right_row["mean_average_reward"])
                            if left_row.get("mean_average_reward") is not None and right_row.get("mean_average_reward") is not None
                            else None
                        ),
                        "outcome": outcome,
                    }
                )
    return comparisons


def build_concentration_summary(problem_host_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in problem_host_rows:
        family_shares = [float(row.get(f"mean_family_share_{label}") or 0.0) for label in FAMILY_LABELS]
        regime_shares = [float(row.get(f"mean_regime_share_{label}") or 0.0) for label in REGIME_LABELS]
        intent_shares = [float(row.get(f"mean_intent_share_{label}") or 0.0) for label in INTENT_LABELS]
        rows.append(
            {
                "suite": row.get("suite"),
                "host": row["host"],
                "problem": row["problem"],
                "variant": row["variant"],
                "variant_group": row.get("variant_group"),
                "dominant_family_share": max(family_shares) if family_shares else 0.0,
                "dominant_regime_share": max(regime_shares) if regime_shares else 0.0,
                "dominant_intent_share": max(intent_shares) if intent_shares else 0.0,
                "family_concentration": row.get("mean_family_concentration"),
                "regime_concentration": row.get("mean_regime_concentration"),
                "intent_concentration": row.get("mean_intent_concentration"),
                "family_entropy": _entropy_from_shares(family_shares),
                "regime_entropy": _entropy_from_shares(regime_shares),
                "intent_entropy": _entropy_from_shares(intent_shares),
                "mean_family_switches": row.get("mean_family_switches"),
                "mean_regime_switches": row.get("mean_regime_switches"),
                "mean_intent_switches": row.get("mean_intent_switches"),
            }
        )
    return rows


def build_phase_summary(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not trace_rows:
        return []

    enriched = [dict(row, phase=phase_from_progress(row.get("budget_progress"))) for row in trace_rows]
    switch_groups: dict[tuple[str, str, str, str, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        run_key = (
            str(row.get("host")),
            str(row.get("problem")),
            str(row.get("variant")),
            str(row.get("phase")),
            row.get("run_id"),
        )
        switch_groups[run_key].append(row)

    per_run_phase_switches: dict[tuple[str, str, str, str], list[dict[str, float]]] = defaultdict(list)
    for (host, problem, variant, phase, _run_id), rows in switch_groups.items():
        ordered = sorted(rows, key=lambda item: float(item.get("step_index") or 0.0))
        family_switches = 0
        regime_switches = 0
        intent_switches = 0
        for prev, cur in zip(ordered, ordered[1:]):
            if prev.get("operator_family") != cur.get("operator_family"):
                family_switches += 1
            if prev.get("regime") != cur.get("regime"):
                regime_switches += 1
            if prev.get("intent_prototype") != cur.get("intent_prototype"):
                intent_switches += 1
        per_run_phase_switches[(host, problem, variant, phase)].append(
            {
                "family_switches": float(family_switches),
                "regime_switches": float(regime_switches),
                "intent_switches": float(intent_switches),
            }
        )

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        grouped[(str(row.get("host")), str(row.get("problem")), str(row.get("variant")), str(row.get("phase")))].append(row)

    output: list[dict[str, Any]] = []
    for phase_key, rows in sorted(grouped.items()):
        host, problem, variant, phase = phase_key
        reward = [float(row.get("bounded_reward") or 0.0) for row in rows]
        overhead = [float(row.get("overhead_ms") or 0.0) for row in rows if row.get("overhead_ms") is not None]
        counts_family: defaultdict[str, float] = defaultdict(float)
        counts_regime: defaultdict[str, float] = defaultdict(float)
        counts_intent: defaultdict[str, float] = defaultdict(float)
        for row in rows:
            counts_family[str(row.get("operator_family"))] += 1.0
            counts_regime[str(row.get("regime"))] += 1.0
            counts_intent[str(row.get("intent_prototype"))] += 1.0
        total = float(len(rows))
        payload: dict[str, Any] = {
            "suite": rows[0].get("suite"),
            "host": host,
            "problem": problem,
            "variant": variant,
            "phase": phase,
            "n_steps": len(rows),
            "mean_reward": mean(reward),
            "mean_overhead_ms": mean(overhead),
            "mean_family_switches": mean([item["family_switches"] for item in per_run_phase_switches[phase_key]]),
            "mean_regime_switches": mean([item["regime_switches"] for item in per_run_phase_switches[phase_key]]),
            "mean_intent_switches": mean([item["intent_switches"] for item in per_run_phase_switches[phase_key]]),
        }
        for label in FAMILY_LABELS:
            payload[f"family_share_{label}"] = counts_family[label] / total if total > 0 else 0.0
        for label in REGIME_LABELS:
            payload[f"regime_share_{label}"] = counts_regime[label] / total if total > 0 else 0.0
        for label in INTENT_LABELS:
            payload[f"intent_share_{label}"] = counts_intent[label] / total if total > 0 else 0.0
        output.append(payload)
    return output


__all__ = [
    "ADAPTIVE_VARIANTS",
    "BASELINE_BY_HOST",
    "INTENT_LABELS",
    "build_analysis_report",
    "build_concentration_summary",
    "build_go_no_go_analysis",
    "build_heterogeneity_summary",
    "build_phase_summary",
    "build_policy_comparison",
    "compute_problem_host_summary",
    "load_pilot_output",
    "phase_from_progress",
    "read_csv_rows",
    "write_csv_rows",
]
