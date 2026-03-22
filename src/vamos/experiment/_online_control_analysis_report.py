from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

ADAPTIVE_VARIANTS = ("adaptive_flat_operator", "adaptive_flat_parameter", "adaptive_hierarchical_joint")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


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
    for group_key, group in grouped.items():
        winner_row = max(group, key=lambda item: float(item.get("mean_hv") or 0.0))
        adaptive_winners[group_key] = str(winner_row["variant"])
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
    for (_host, problem), winner_variant in adaptive_winners.items():
        problem_to_host_winners[problem].add(winner_variant)
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
        family_labels = [key.removeprefix("family_share_") for key in early if str(key).startswith("family_share_")]
        intent_labels = [key.removeprefix("intent_share_") for key in early if str(key).startswith("intent_share_")]
        family_shift = 0.5 * sum(
            abs(float(early.get(f"family_share_{label}") or 0.0) - float(late.get(f"family_share_{label}") or 0.0))
            for label in family_labels
        )
        intent_shift = 0.5 * sum(
            abs(float(early.get(f"intent_share_{label}") or 0.0) - float(late.get(f"intent_share_{label}") or 0.0))
            for label in intent_labels
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
            "adaptive_hierarchical_joint_mean_hv_gap": _mean(
                [float(row.get("hv_gap_vs_best_fixed") or 0.0) for row in hierarchical_rows if row.get("hv_gap_vs_best_fixed") is not None]
            ),
        },
        "hierarchy_vs_flat": {
            "wins": sum(1 for row in hier_vs_flat if row.get("outcome") == "win"),
            "losses": sum(1 for row in hier_vs_flat if row.get("outcome") == "loss"),
            "ties": sum(1 for row in hier_vs_flat if row.get("outcome") == "tie"),
            "mean_hv_gap": _mean([float(row.get("hv_gap") or 0.0) for row in hier_vs_flat]),
        },
        "concentration": {
            "mean_dominant_family_share": _mean([float(row.get("dominant_family_share") or 0.0) for row in concentration_adaptive]),
            "mean_dominant_intent_share": _mean([float(row.get("dominant_intent_share") or 0.0) for row in concentration_adaptive]),
        },
        "heterogeneity": {
            "phase_family_shift_mean": _mean(
                [float(row.get("value") or 0.0) for row in heterogeneity_rows if row.get("metric") == "phase_family_shift_tvd"]
            ),
            "phase_intent_shift_mean": _mean(
                [float(row.get("value") or 0.0) for row in heterogeneity_rows if row.get("metric") == "phase_intent_shift_tvd"]
            ),
            "phase_reward_range_mean": _mean(
                [float(row.get("value") or 0.0) for row in heterogeneity_rows if row.get("metric") == "phase_reward_range"]
            ),
        },
        "transfer": {
            "available": bool(transfer_rows),
            "mean_hv_delta_warm_vs_cold": _mean(
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
    mean_dominant_family_share = _mean([float(row.get("dominant_family_share") or 0.0) for row in adaptive_concentration])

    phase_family_shift_mean = _mean(
        [float(row.get("value") or 0.0) for row in heterogeneity_rows if row.get("metric") == "phase_family_shift_tvd"]
    )
    best_fixed_variability = [
        row for row in heterogeneity_rows if row.get("metric") == "best_fixed_winner_variability"
    ]
    best_fixed_variability_value = float(best_fixed_variability[0]["value"]) if best_fixed_variability else 0.0

    transfer_rows = transfer_summary or []
    transfer_mean_hv_delta = _mean(
        [float(row.get("hv_delta_warm_vs_cold") or 0.0) for row in transfer_rows if row.get("hv_delta_warm_vs_cold") is not None]
    )
    transfer_helpful = bool(transfer_rows) and transfer_mean_hv_delta >= 0.0

    checks: dict[str, dict[str, Any]] = {
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

    core_passes = sum(
        1
        for key, value in checks.items()
        if key != "warm_start_transfer_non_harmful" and bool(value.get("passed"))
    )
    overall = "go" if core_passes >= 4 and bool(checks["warm_start_transfer_non_harmful"].get("passed", False)) else "caution"
    return {
        "overall_decision": overall,
        "checks": checks,
    }


__all__ = [
    "build_analysis_report",
    "build_go_no_go_analysis",
    "build_heterogeneity_summary",
]
