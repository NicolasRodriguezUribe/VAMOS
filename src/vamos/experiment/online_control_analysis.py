from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from vamos.engine.adaptation.online_control import DEFAULT_PROTOTYPE_SET, OperatorFamily, Regime, available_intent_prototypes

FAMILY_LABELS = tuple(family.value for family in OperatorFamily)
REGIME_LABELS = tuple(regime.value for regime in Regime)
INTENT_LABELS = available_intent_prototypes(DEFAULT_PROTOTYPE_SET)
ADAPTIVE_VARIANTS = ("adaptive_flat_operator", "adaptive_flat_parameter", "adaptive_hierarchical_joint")
BASELINE_BY_HOST = {"nsgaii": "fixed_sbx", "moead": "fixed_de"}


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
            float(row.get("mean_hv") or 0.0),
            -float(row.get("mean_igd_plus")) if row.get("mean_igd_plus") is not None else float("-inf"),
            -float(row.get("mean_time_ms") or 0.0),
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
        key = (
            str(row.get("host")),
            str(row.get("problem")),
            str(row.get("variant")),
            str(row.get("phase")),
            row.get("run_id"),
        )
        switch_groups[key].append(row)

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
    for key, rows in sorted(grouped.items()):
        host, problem, variant, phase = key
        reward = [float(row.get("bounded_reward") or 0.0) for row in rows]
        overhead = [float(row.get("overhead_ms") or 0.0) for row in rows if row.get("overhead_ms") is not None]
        counts_family = defaultdict(float)
        counts_regime = defaultdict(float)
        counts_intent = defaultdict(float)
        for row in rows:
            counts_family[str(row.get("operator_family"))] += 1.0
            counts_regime[str(row.get("regime"))] += 1.0
            counts_intent[str(row.get("intent_prototype"))] += 1.0
        total = float(len(rows))
        payload: dict[str, Any] = {
            "host": host,
            "problem": problem,
            "variant": variant,
            "phase": phase,
            "n_steps": len(rows),
            "mean_reward": mean(reward),
            "mean_overhead_ms": mean(overhead),
            "mean_family_switches": mean([item["family_switches"] for item in per_run_phase_switches[key]]),
            "mean_regime_switches": mean([item["regime_switches"] for item in per_run_phase_switches[key]]),
            "mean_intent_switches": mean([item["intent_switches"] for item in per_run_phase_switches[key]]),
        }
        for label in FAMILY_LABELS:
            payload[f"family_share_{label}"] = counts_family[label] / total if total > 0 else 0.0
        for label in REGIME_LABELS:
            payload[f"regime_share_{label}"] = counts_regime[label] / total if total > 0 else 0.0
        for label in INTENT_LABELS:
            payload[f"intent_share_{label}"] = counts_intent[label] / total if total > 0 else 0.0
        output.append(payload)
    return output


def build_heterogeneity_summary(problem_host_rows: list[dict[str, Any]], phase_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    best_fixed_variants = {
        (str(row["host"]), str(row["problem"])): str(row["best_fixed_variant"])
        for row in problem_host_rows
        if row.get("comparison_to_best_fixed") == "best_fixed" and row.get("best_fixed_variant") is not None
    }
    if best_fixed_variants:
        unique_winners = sorted(set(best_fixed_variants.values()))
        rows.append(
            {
                "metric": "best_fixed_winner_variability",
                "scope": "overall",
                "value": float(len(unique_winners)),
                "details": {"winners": unique_winners, "cases": len(best_fixed_variants)},
            }
        )

    adaptive_winners: dict[tuple[str, str], str] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in problem_host_rows:
        if row.get("variant") in ADAPTIVE_VARIANTS:
            grouped[(str(row["host"]), str(row["problem"]))].append(row)
    for key, group in grouped.items():
        winner = max(group, key=lambda item: float(item.get("mean_hv") or 0.0))
        adaptive_winners[key] = str(winner["variant"])
    if adaptive_winners:
        unique_winners = sorted(set(adaptive_winners.values()))
        rows.append(
            {
                "metric": "adaptive_winner_variability",
                "scope": "overall",
                "value": float(len(unique_winners)),
                "details": {"winners": unique_winners, "cases": len(adaptive_winners)},
            }
        )

    problem_to_host_winners: dict[str, set[str]] = defaultdict(set)
    for (host, problem), winner in adaptive_winners.items():
        problem_to_host_winners[problem].add(winner)
    for problem, winners in sorted(problem_to_host_winners.items()):
        rows.append(
            {
                "metric": "host_dependent_adaptive_winner",
                "scope": "problem",
                "problem": problem,
                "value": 1.0 if len(winners) > 1 else 0.0,
                "details": {"winners": sorted(winners)},
            }
        )

    phase_grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in phase_rows:
        phase_grouped[(str(row["host"]), str(row["problem"]), str(row["variant"]))][str(row["phase"])] = row
    for (host, problem, variant), phases in sorted(phase_grouped.items()):
        if "early" not in phases or "late" not in phases:
            continue
        early = phases["early"]
        late = phases["late"]
        family_shift = 0.5 * sum(
            abs(float(early.get(f"family_share_{label}") or 0.0) - float(late.get(f"family_share_{label}") or 0.0))
            for label in FAMILY_LABELS
        )
        intent_shift = 0.5 * sum(
            abs(float(early.get(f"intent_share_{label}") or 0.0) - float(late.get(f"intent_share_{label}") or 0.0))
            for label in INTENT_LABELS
        )
        reward_values = [float(item.get("mean_reward") or 0.0) for item in phases.values()]
        rows.extend(
            [
                {
                    "metric": "phase_family_shift_tvd",
                    "scope": "run_group",
                    "host": host,
                    "problem": problem,
                    "variant": variant,
                    "value": family_shift,
                },
                {
                    "metric": "phase_intent_shift_tvd",
                    "scope": "run_group",
                    "host": host,
                    "problem": problem,
                    "variant": variant,
                    "value": intent_shift,
                },
                {
                    "metric": "phase_reward_range",
                    "scope": "run_group",
                    "host": host,
                    "problem": problem,
                    "variant": variant,
                    "value": max(reward_values) - min(reward_values),
                },
            ]
        )
    return rows


def build_analysis_report(
    problem_host_rows: list[dict[str, Any]],
    policy_comparison: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    heterogeneity_rows: list[dict[str, Any]],
    phase_rows: list[dict[str, Any]],
    transfer_summary: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    hierarchical_rows = [row for row in problem_host_rows if row.get("variant") == "adaptive_hierarchical_joint"]
    hierarchical_wins = sum(1 for row in hierarchical_rows if row.get("comparison_to_best_fixed") == "win")
    hierarchical_cases = len(hierarchical_rows)
    hier_vs_flat = [
        row for row in policy_comparison if row.get("left_variant") == "adaptive_hierarchical_joint" and row.get("right_variant") in ADAPTIVE_VARIANTS
    ]
    concentration_adaptive = [row for row in concentration_rows if str(row.get("variant", "")).startswith("adaptive_")]
    transfer_rows = transfer_summary or []
    return {
        "counts": {
            "problem_host_rows": len(problem_host_rows),
            "policy_comparison_rows": len(policy_comparison),
            "concentration_rows": len(concentration_rows),
            "heterogeneity_rows": len(heterogeneity_rows),
            "phase_rows": len(phase_rows),
            "transfer_rows": len(transfer_rows),
        },
        "best_fixed": {
            "adaptive_hierarchical_joint_wins": hierarchical_wins,
            "adaptive_hierarchical_joint_cases": hierarchical_cases,
            "adaptive_hierarchical_joint_mean_hv_gap": mean(
                [float(row.get("hv_gap_vs_best_fixed") or 0.0) for row in hierarchical_rows if row.get("hv_gap_vs_best_fixed") is not None]
            ),
        },
        "hierarchy_vs_flat": {
            "wins": sum(1 for row in hier_vs_flat if row.get("outcome") == "win"),
            "losses": sum(1 for row in hier_vs_flat if row.get("outcome") == "loss"),
            "ties": sum(1 for row in hier_vs_flat if row.get("outcome") == "tie"),
            "mean_hv_gap": mean([float(row.get("hv_gap") or 0.0) for row in hier_vs_flat]),
        },
        "concentration": {
            "mean_dominant_family_share": mean([float(row.get("dominant_family_share") or 0.0) for row in concentration_adaptive]),
            "mean_dominant_intent_share": mean([float(row.get("dominant_intent_share") or 0.0) for row in concentration_adaptive]),
        },
        "heterogeneity": {
            "phase_family_shift_mean": mean(
                [float(row.get("value") or 0.0) for row in heterogeneity_rows if row.get("metric") == "phase_family_shift_tvd"]
            ),
            "phase_intent_shift_mean": mean(
                [float(row.get("value") or 0.0) for row in heterogeneity_rows if row.get("metric") == "phase_intent_shift_tvd"]
            ),
            "phase_reward_range_mean": mean(
                [float(row.get("value") or 0.0) for row in heterogeneity_rows if row.get("metric") == "phase_reward_range"]
            ),
        },
        "transfer": {
            "available": bool(transfer_rows),
            "mean_hv_delta_warm_vs_cold": mean(
                [float(row.get("hv_delta_warm_vs_cold") or 0.0) for row in transfer_rows if row.get("hv_delta_warm_vs_cold") is not None]
            ),
        },
    }


def build_go_no_go_analysis(
    problem_host_rows: list[dict[str, Any]],
    policy_comparison: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    heterogeneity_rows: list[dict[str, Any]],
    transfer_summary: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    hierarchical_rows = [row for row in problem_host_rows if row.get("variant") == "adaptive_hierarchical_joint"]
    beat_best_fixed_cases = sum(1 for row in hierarchical_rows if row.get("comparison_to_best_fixed") == "win")
    comparable_best_fixed = len(hierarchical_rows)
    threshold_best_fixed = max(1, math.ceil(max(1, comparable_best_fixed) / 3))

    grouped_pairwise: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in policy_comparison:
        if row.get("left_variant") == "adaptive_hierarchical_joint":
            grouped_pairwise[(str(row["host"]), str(row["problem"]))][str(row["right_variant"])] = row
    hierarchical_beats_both = 0
    comparable_pairwise = 0
    for rows in grouped_pairwise.values():
        if "adaptive_flat_operator" in rows and "adaptive_flat_parameter" in rows:
            comparable_pairwise += 1
            if rows["adaptive_flat_operator"]["outcome"] == "win" and rows["adaptive_flat_parameter"]["outcome"] == "win":
                hierarchical_beats_both += 1
    threshold_pairwise = max(1, math.ceil(max(1, comparable_pairwise) / 3))

    runtime_ratios = [float(row.get("runtime_ratio_vs_best_fixed") or 0.0) for row in hierarchical_rows if row.get("runtime_ratio_vs_best_fixed") is not None]
    median_runtime_ratio = sorted(runtime_ratios)[len(runtime_ratios) // 2] if runtime_ratios else None

    adaptive_concentration = [row for row in concentration_rows if str(row.get("variant", "")).startswith("adaptive_")]
    mean_dominant_family_share = mean([float(row.get("dominant_family_share") or 0.0) for row in adaptive_concentration])

    phase_family_shift_mean = mean(
        [float(row.get("value") or 0.0) for row in heterogeneity_rows if row.get("metric") == "phase_family_shift_tvd"]
    )
    best_fixed_variability = [
        row for row in heterogeneity_rows if row.get("metric") == "best_fixed_winner_variability"
    ]
    best_fixed_variability_value = float(best_fixed_variability[0]["value"]) if best_fixed_variability else 0.0

    transfer_rows = transfer_summary or []
    transfer_mean_hv_delta = mean(
        [float(row.get("hv_delta_warm_vs_cold") or 0.0) for row in transfer_rows if row.get("hv_delta_warm_vs_cold") is not None]
    )
    transfer_helpful = bool(transfer_rows) and transfer_mean_hv_delta >= 0.0

    checks = {
        "adaptive_hierarchical_beats_best_fixed": {
            "passed": beat_best_fixed_cases >= threshold_best_fixed,
            "wins": beat_best_fixed_cases,
            "cases": comparable_best_fixed,
            "threshold": threshold_best_fixed,
        },
        "adaptive_hierarchical_beats_both_flat_adaptive": {
            "passed": hierarchical_beats_both >= threshold_pairwise,
            "wins": hierarchical_beats_both,
            "cases": comparable_pairwise,
            "threshold": threshold_pairwise,
        },
        "runtime_overhead_below_threshold": {
            "passed": median_runtime_ratio is not None and median_runtime_ratio <= 1.25,
            "median_runtime_ratio": median_runtime_ratio,
            "threshold": 1.25,
        },
        "dynamic_heterogeneity_present": {
            "passed": best_fixed_variability_value > 1.0 or phase_family_shift_mean >= 0.10,
            "best_fixed_winner_count": best_fixed_variability_value,
            "phase_family_shift_mean": phase_family_shift_mean,
            "threshold_phase_family_shift": 0.10,
        },
        "adaptive_concentration_above_minimum": {
            "passed": mean_dominant_family_share >= 0.55,
            "mean_dominant_family_share": mean_dominant_family_share,
            "threshold": 0.55,
        },
        "warm_start_transfer_non_harmful": {
            "available": bool(transfer_rows),
            "passed": transfer_helpful,
            "mean_hv_delta_warm_vs_cold": transfer_mean_hv_delta,
        },
    }

    core_passes = sum(1 for key, value in checks.items() if key != "warm_start_transfer_non_harmful" and value.get("passed"))
    overall = "go" if core_passes >= 4 and checks["warm_start_transfer_non_harmful"].get("passed", False) else "caution"
    return {
        "overall_decision": overall,
        "checks": checks,
    }


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
