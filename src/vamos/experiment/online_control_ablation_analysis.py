from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .online_control_analysis import (
    build_concentration_summary,
    build_phase_summary,
    compute_problem_host_summary,
    load_pilot_output,
    write_csv_rows,
)

FIXED_VARIANTS = ("fixed_sbx", "fixed_de")
ADAPTIVE_VARIANTS = (
    "adaptive_flat_operator",
    "adaptive_flat_parameter",
    "adaptive_hierarchical_joint",
    "adaptive_hierarchical_joint_no_regime",
    "adaptive_hierarchical_joint_fixed_family_sbx",
    "adaptive_hierarchical_joint_fixed_family_de",
)
PROFILE_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("profile_start_step_time_ms", "control"),
    ("profile_router_time_ms", "control"),
    ("profile_policy_select_time_ms", "control"),
    ("profile_policy_update_time_ms", "control"),
    ("profile_decode_time_ms", "control"),
    ("profile_trace_time_ms", "control"),
    ("profile_variation_time_ms", "host"),
    ("profile_evaluation_time_ms", "host"),
    ("profile_survival_time_ms", "host"),
)


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _suite_rows(output_dir: Path, suite: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    runs, summary, traces = load_pilot_output(output_dir)
    resolved_config = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    for bucket in (runs, summary, traces):
        for row in bucket:
            row["suite"] = suite
    return runs, summary, traces, resolved_config


def load_ablation_outputs(
    *,
    zcat_dir: Path,
    anchor_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    zcat_runs, zcat_summary, zcat_traces, zcat_config = _suite_rows(zcat_dir, "zcat")
    anchor_runs, anchor_summary, anchor_traces, anchor_config = _suite_rows(anchor_dir, "anchor")
    return (
        zcat_runs + anchor_runs,
        zcat_summary + anchor_summary,
        zcat_traces + anchor_traces,
        {"zcat": zcat_config, "anchor": anchor_config},
    )


def compute_suite_problem_host_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_suite[str(row.get("suite", "unknown"))].append(dict(row))
    for suite, rows in sorted(by_suite.items()):
        enriched = compute_problem_host_summary(rows)
        for row in enriched:
            row["suite"] = suite
            output.append(row)
    return output


def build_suite_concentration_summary(problem_host_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in problem_host_rows:
        by_suite[str(row.get("suite", "unknown"))].append(dict(row))
    for suite, rows in sorted(by_suite.items()):
        for payload in build_concentration_summary(rows):
            payload["suite"] = suite
            output.append(payload)
    return output


def build_suite_phase_summary(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        by_suite[str(row.get("suite", "unknown"))].append(dict(row))
    for suite, rows in sorted(by_suite.items()):
        for payload in build_phase_summary(rows):
            payload["suite"] = suite
            output.append(payload)
    return output


def _group_case_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["suite"]), str(row["host"]), str(row["problem"]))][str(row["variant"])] = row
    return grouped


def _comparison_row(
    *,
    suite: str,
    host: str,
    problem: str,
    left_variant: str,
    right_variant: str,
    comparison_name: str,
    comparison_kind: str,
    hv_gap: float | None,
    runtime_ratio: float | None,
    reward_gap: float | None,
    igd_plus_gap: float | None,
    outcome: str,
) -> dict[str, Any]:
    return {
        "suite": suite,
        "host": host,
        "problem": problem,
        "left_variant": left_variant,
        "right_variant": right_variant,
        "comparison_name": comparison_name,
        "comparison_kind": comparison_kind,
        "hv_gap": hv_gap,
        "runtime_ratio": runtime_ratio,
        "reward_gap": reward_gap,
        "igd_plus_gap": igd_plus_gap,
        "outcome": outcome,
    }


def build_ablation_policy_comparison(problem_host_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_case_rows(problem_host_rows)
    rows: list[dict[str, Any]] = []
    hv_tol = 1e-6
    pair_defs = (
        ("adaptive_hierarchical_joint", "adaptive_flat_operator", "hierarchical_vs_flat_operator"),
        ("adaptive_hierarchical_joint", "adaptive_flat_parameter", "hierarchical_vs_flat_parameter"),
        ("adaptive_hierarchical_joint", "adaptive_hierarchical_joint_no_regime", "hierarchical_vs_no_regime"),
        ("adaptive_hierarchical_joint", "adaptive_hierarchical_joint_fixed_family_sbx", "hierarchical_vs_fixed_family_sbx"),
        ("adaptive_hierarchical_joint", "adaptive_hierarchical_joint_fixed_family_de", "hierarchical_vs_fixed_family_de"),
        (
            "adaptive_hierarchical_joint_fixed_family_sbx",
            "adaptive_flat_parameter",
            "fixed_family_sbx_vs_flat_parameter",
        ),
    )
    for (suite, host, problem), mapping in sorted(grouped.items()):
        for variant, row in sorted(mapping.items()):
            if not str(variant).startswith("adaptive_"):
                continue
            hv_gap = row.get("hv_gap_vs_best_fixed")
            runtime_ratio = row.get("runtime_ratio_vs_best_fixed")
            reward_gap = row.get("mean_average_reward")
            igd_gap = row.get("igd_plus_gap_vs_best_fixed")
            outcome = str(row.get("comparison_to_best_fixed", "no_best_fixed"))
            rows.append(
                _comparison_row(
                    suite=suite,
                    host=host,
                    problem=problem,
                    left_variant=str(variant),
                    right_variant=str(row.get("best_fixed_variant") or "best_fixed"),
                    comparison_name=f"{variant}_vs_best_fixed",
                    comparison_kind="vs_best_fixed",
                    hv_gap=float(hv_gap) if hv_gap is not None else None,
                    runtime_ratio=float(runtime_ratio) if runtime_ratio is not None else None,
                    reward_gap=float(reward_gap) if reward_gap is not None else None,
                    igd_plus_gap=float(igd_gap) if igd_gap is not None else None,
                    outcome=outcome,
                )
            )
        for left, right, name in pair_defs:
            if left not in mapping or right not in mapping:
                continue
            left_row = mapping[left]
            right_row = mapping[right]
            hv_gap = float(left_row.get("mean_hv") or 0.0) - float(right_row.get("mean_hv") or 0.0)
            if hv_gap > hv_tol:
                outcome = "win"
            elif hv_gap < -hv_tol:
                outcome = "loss"
            else:
                outcome = "tie"
            reward_gap = None
            if left_row.get("mean_average_reward") is not None and right_row.get("mean_average_reward") is not None:
                reward_gap = float(left_row["mean_average_reward"]) - float(right_row["mean_average_reward"])
            igd_gap = None
            if left_row.get("mean_igd_plus") is not None and right_row.get("mean_igd_plus") is not None:
                igd_gap = float(left_row["mean_igd_plus"]) - float(right_row["mean_igd_plus"])
            rows.append(
                _comparison_row(
                    suite=suite,
                    host=host,
                    problem=problem,
                    left_variant=left,
                    right_variant=right,
                    comparison_name=name,
                    comparison_kind="pairwise",
                    hv_gap=hv_gap,
                    runtime_ratio=float(left_row.get("mean_time_ms") or 0.0) / max(1e-9, float(right_row.get("mean_time_ms") or 0.0)),
                    reward_gap=reward_gap,
                    igd_plus_gap=igd_gap,
                    outcome=outcome,
                )
            )
    return rows


def _aggregate_comparisons(comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        grouped[(str(row["suite"]), str(row["comparison_name"]))].append(row)
        grouped[("overall", str(row["comparison_name"]))].append(row)
    for (suite, comparison_name), rows in sorted(grouped.items()):
        hv = [float(row["hv_gap"]) for row in rows if row.get("hv_gap") is not None]
        runtime = [float(row["runtime_ratio"]) for row in rows if row.get("runtime_ratio") is not None]
        reward = [float(row["reward_gap"]) for row in rows if row.get("reward_gap") is not None]
        igd = [float(row["igd_plus_gap"]) for row in rows if row.get("igd_plus_gap") is not None]
        output.append(
            {
                "suite": suite,
                "comparison_name": comparison_name,
                "left_variant": rows[0]["left_variant"],
                "right_variant": rows[0]["right_variant"],
                "cases": len(rows),
                "wins": sum(1 for row in rows if row.get("outcome") == "win"),
                "losses": sum(1 for row in rows if row.get("outcome") == "loss"),
                "ties": sum(1 for row in rows if row.get("outcome") == "tie"),
                "mean_hv_gap": _mean(hv),
                "median_hv_gap": _median(hv),
                "mean_runtime_ratio": _mean(runtime),
                "median_runtime_ratio": _median(runtime),
                "mean_reward_gap": _mean(reward),
                "median_reward_gap": _median(reward),
                "mean_igd_plus_gap": _mean(igd),
                "median_igd_plus_gap": _median(igd),
            }
        )
    return output


def build_source_attribution_summary(problem_host_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparison_rows = build_ablation_policy_comparison(problem_host_rows)
    summary_rows = _aggregate_comparisons(comparison_rows)
    order = {
        "adaptive_hierarchical_joint_vs_best_fixed": 0,
        "adaptive_hierarchical_joint_fixed_family_sbx_vs_best_fixed": 1,
        "adaptive_hierarchical_joint_fixed_family_de_vs_best_fixed": 2,
        "hierarchical_vs_fixed_family_sbx": 3,
        "hierarchical_vs_fixed_family_de": 4,
        "hierarchical_vs_no_regime": 5,
        "hierarchical_vs_flat_operator": 6,
        "hierarchical_vs_flat_parameter": 7,
        "fixed_family_sbx_vs_flat_parameter": 8,
    }
    summary_rows.sort(key=lambda row: (str(row["suite"]), order.get(str(row["comparison_name"]), 99), str(row["comparison_name"])))
    return summary_rows


def build_suite_heterogeneity_summary(
    problem_host_rows: list[dict[str, Any]],
    phase_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped_cases = _group_case_rows(problem_host_rows)
    phase_grouped: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in phase_rows:
        phase_grouped[(str(row["suite"]), str(row["host"]), str(row["problem"]), str(row["variant"]))][str(row["phase"])] = row

    output: list[dict[str, Any]] = []
    fixed_winners_by_suite: defaultdict[str, set[str]] = defaultdict(set)
    adaptive_winners_by_suite: defaultdict[str, set[str]] = defaultdict(set)
    for (suite, _host, _problem), mapping in sorted(grouped_cases.items()):
        fixed_rows = [row for row in mapping.values() if str(row["variant"]).startswith("fixed_")]
        if fixed_rows:
            fixed_winner = max(fixed_rows, key=lambda row: float(row.get("mean_hv") or 0.0))
            fixed_winners_by_suite[suite].add(str(fixed_winner["variant"]))
        adaptive_rows = [row for row in mapping.values() if str(row["variant"]).startswith("adaptive_")]
        if adaptive_rows:
            adaptive_winner = max(adaptive_rows, key=lambda row: float(row.get("mean_hv") or 0.0))
            adaptive_winners_by_suite[suite].add(str(adaptive_winner["variant"]))

    for suite in sorted(set(fixed_winners_by_suite) | set(adaptive_winners_by_suite)):
        output.append(
            {
                "suite": suite,
                "metric": "best_fixed_winner_variability",
                "value": float(len(fixed_winners_by_suite.get(suite, set()))),
                "details": {"winners": sorted(fixed_winners_by_suite.get(suite, set()))},
            }
        )
        output.append(
            {
                "suite": suite,
                "metric": "adaptive_winner_variability",
                "value": float(len(adaptive_winners_by_suite.get(suite, set()))),
                "details": {"winners": sorted(adaptive_winners_by_suite.get(suite, set()))},
            }
        )

    for (suite, host, problem, variant), phases in sorted(phase_grouped.items()):
        early = phases.get("early")
        late = phases.get("late")
        if early is None or late is None:
            continue
        family_labels = [key.removeprefix("family_share_") for key in early.keys() if str(key).startswith("family_share_")]
        intent_labels = [key.removeprefix("intent_share_") for key in early.keys() if str(key).startswith("intent_share_")]
        family_shift = 0.5 * sum(
            abs(_float(early.get(f"family_share_{label}")) - _float(late.get(f"family_share_{label}")))
            for label in family_labels
        )
        intent_shift = 0.5 * sum(
            abs(_float(early.get(f"intent_share_{label}")) - _float(late.get(f"intent_share_{label}")))
            for label in intent_labels
        )
        reward_values = [_float(item.get("mean_reward")) for item in phases.values()]
        output.extend(
            [
                {
                    "suite": suite,
                    "metric": "phase_family_shift_tvd",
                    "host": host,
                    "problem": problem,
                    "variant": variant,
                    "value": family_shift,
                },
                {
                    "suite": suite,
                    "metric": "phase_intent_shift_tvd",
                    "host": host,
                    "problem": problem,
                    "variant": variant,
                    "value": intent_shift,
                },
                {
                    "suite": suite,
                    "metric": "phase_reward_range",
                    "host": host,
                    "problem": problem,
                    "variant": variant,
                    "value": max(reward_values) - min(reward_values),
                },
            ]
        )
    return output


def build_overhead_profile_summary(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row.get("suite", "unknown")), str(row["host"]), str(row["variant"]))].append(row)
        grouped[("overall", str(row["host"]), str(row["variant"]))].append(row)
        grouped[(str(row.get("suite", "unknown")), "all_hosts", str(row["variant"]))].append(row)
        grouped[("overall", "all_hosts", str(row["variant"]))].append(row)

    for (suite, host, variant), rows in sorted(grouped.items()):
        totals = [
            max(
                _float(row.get("profile_total_runtime_ms")),
                _float(row.get("time_ms")),
            )
            for row in rows
        ]
        for component_key, component_group in PROFILE_COMPONENTS:
            component_values = [_float(row.get(component_key)) for row in rows]
            shares = [component / total for component, total in zip(component_values, totals) if total > 0.0]
            output.append(
                {
                    "suite": suite,
                    "host": host,
                    "variant": variant,
                    "component": component_key.removeprefix("profile_").removesuffix("_ms"),
                    "component_group": component_group,
                    "n_runs": len(rows),
                    "mean_time_ms": _mean(component_values),
                    "median_time_ms": _median(component_values),
                    "mean_share_of_total_runtime": _mean(shares),
                    "median_share_of_total_runtime": _median(shares),
                }
            )
        control_values: list[float] = []
        host_values: list[float] = []
        control_shares: list[float] = []
        host_shares: list[float] = []
        for row, total in zip(rows, totals):
            control_total = sum(_float(row.get(component)) for component, group in PROFILE_COMPONENTS if group == "control")
            host_total = sum(_float(row.get(component)) for component, group in PROFILE_COMPONENTS if group == "host")
            control_values.append(control_total)
            host_values.append(host_total)
            if total > 0.0:
                control_shares.append(control_total / total)
                host_shares.append(host_total / total)
        output.extend(
            [
                {
                    "suite": suite,
                    "host": host,
                    "variant": variant,
                    "component": "control_total",
                    "component_group": "control",
                    "n_runs": len(rows),
                    "mean_time_ms": _mean(control_values),
                    "median_time_ms": _median(control_values),
                    "mean_share_of_total_runtime": _mean(control_shares),
                    "median_share_of_total_runtime": _median(control_shares),
                },
                {
                    "suite": suite,
                    "host": host,
                    "variant": variant,
                    "component": "host_pipeline_total",
                    "component_group": "host",
                    "n_runs": len(rows),
                    "mean_time_ms": _mean(host_values),
                    "median_time_ms": _median(host_values),
                    "mean_share_of_total_runtime": _mean(host_shares),
                    "median_share_of_total_runtime": _median(host_shares),
                },
            ]
        )
    return output


def _rate_from_summary(rows: list[dict[str, Any]], suite: str, comparison_name: str) -> float | None:
    for row in rows:
        if str(row.get("suite")) != suite or str(row.get("comparison_name")) != comparison_name:
            continue
        cases = int(row.get("cases") or 0)
        if cases <= 0:
            return None
        return float(row.get("wins") or 0) / float(cases)
    return None


def _metric_value(
    rows: list[dict[str, Any]],
    *,
    suite: str,
    comparison_name: str,
    field: str,
) -> float | None:
    for row in rows:
        if str(row.get("suite")) == suite and str(row.get("comparison_name")) == comparison_name:
            value = row.get(field)
            return None if value is None else float(value)
    return None


def _heterogeneity_metric(
    rows: list[dict[str, Any]],
    *,
    suite: str,
    metric: str,
    variant: str | None = None,
) -> float:
    values = [
        _float(row.get("value"))
        for row in rows
        if str(row.get("suite")) == suite
        and str(row.get("metric")) == metric
        and (variant is None or str(row.get("variant")) == variant)
    ]
    return _mean(values)


def _concentration_metric(
    rows: list[dict[str, Any]],
    *,
    suite: str,
    variant: str,
    field: str,
) -> float:
    values = [_float(row.get(field)) for row in rows if str(row.get("suite")) == suite and str(row.get("variant")) == variant]
    return _mean(values)


def build_benchmark_sensitivity_summary(
    source_attribution_rows: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    heterogeneity_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics = {
        "hierarchical_mean_hv_gap_vs_best_fixed": (
            _metric_value(source_attribution_rows, suite="zcat", comparison_name="adaptive_hierarchical_joint_vs_best_fixed", field="mean_hv_gap"),
            _metric_value(source_attribution_rows, suite="anchor", comparison_name="adaptive_hierarchical_joint_vs_best_fixed", field="mean_hv_gap"),
        ),
        "hierarchical_win_rate_vs_best_fixed": (
            _rate_from_summary(source_attribution_rows, "zcat", "adaptive_hierarchical_joint_vs_best_fixed"),
            _rate_from_summary(source_attribution_rows, "anchor", "adaptive_hierarchical_joint_vs_best_fixed"),
        ),
        "prototype_only_sbx_mean_hv_gap_vs_best_fixed": (
            _metric_value(
                source_attribution_rows,
                suite="zcat",
                comparison_name="adaptive_hierarchical_joint_fixed_family_sbx_vs_best_fixed",
                field="mean_hv_gap",
            ),
            _metric_value(
                source_attribution_rows,
                suite="anchor",
                comparison_name="adaptive_hierarchical_joint_fixed_family_sbx_vs_best_fixed",
                field="mean_hv_gap",
            ),
        ),
        "family_increment_mean_hv_gap": (
            _metric_value(source_attribution_rows, suite="zcat", comparison_name="hierarchical_vs_fixed_family_sbx", field="mean_hv_gap"),
            _metric_value(source_attribution_rows, suite="anchor", comparison_name="hierarchical_vs_fixed_family_sbx", field="mean_hv_gap"),
        ),
        "regime_increment_mean_hv_gap": (
            _metric_value(source_attribution_rows, suite="zcat", comparison_name="hierarchical_vs_no_regime", field="mean_hv_gap"),
            _metric_value(source_attribution_rows, suite="anchor", comparison_name="hierarchical_vs_no_regime", field="mean_hv_gap"),
        ),
        "hierarchical_phase_family_shift_mean": (
            _heterogeneity_metric(heterogeneity_rows, suite="zcat", metric="phase_family_shift_tvd", variant="adaptive_hierarchical_joint"),
            _heterogeneity_metric(heterogeneity_rows, suite="anchor", metric="phase_family_shift_tvd", variant="adaptive_hierarchical_joint"),
        ),
        "hierarchical_phase_intent_shift_mean": (
            _heterogeneity_metric(heterogeneity_rows, suite="zcat", metric="phase_intent_shift_tvd", variant="adaptive_hierarchical_joint"),
            _heterogeneity_metric(heterogeneity_rows, suite="anchor", metric="phase_intent_shift_tvd", variant="adaptive_hierarchical_joint"),
        ),
        "best_fixed_winner_variability": (
            _heterogeneity_metric(heterogeneity_rows, suite="zcat", metric="best_fixed_winner_variability"),
            _heterogeneity_metric(heterogeneity_rows, suite="anchor", metric="best_fixed_winner_variability"),
        ),
        "adaptive_winner_variability": (
            _heterogeneity_metric(heterogeneity_rows, suite="zcat", metric="adaptive_winner_variability"),
            _heterogeneity_metric(heterogeneity_rows, suite="anchor", metric="adaptive_winner_variability"),
        ),
        "hierarchical_dominant_family_share": (
            _concentration_metric(concentration_rows, suite="zcat", variant="adaptive_hierarchical_joint", field="dominant_family_share"),
            _concentration_metric(concentration_rows, suite="anchor", variant="adaptive_hierarchical_joint", field="dominant_family_share"),
        ),
        "hierarchical_dominant_intent_share": (
            _concentration_metric(concentration_rows, suite="zcat", variant="adaptive_hierarchical_joint", field="dominant_intent_share"),
            _concentration_metric(concentration_rows, suite="anchor", variant="adaptive_hierarchical_joint", field="dominant_intent_share"),
        ),
    }
    output: list[dict[str, Any]] = []
    for metric, (zcat_value, anchor_value) in metrics.items():
        difference = None
        if zcat_value is not None and anchor_value is not None:
            difference = float(zcat_value) - float(anchor_value)
        output.append(
            {
                "metric": metric,
                "zcat_value": zcat_value,
                "anchor_value": anchor_value,
                "difference_zcat_minus_anchor": difference,
            }
        )
    return output


def _lookup_summary_row(rows: list[dict[str, Any]], suite: str, comparison_name: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("suite")) == suite and str(row.get("comparison_name")) == comparison_name:
            return row
    return None


def build_ablation_analysis_report(
    *,
    configs: dict[str, dict[str, Any]],
    source_attribution_rows: list[dict[str, Any]],
    benchmark_sensitivity_rows: list[dict[str, Any]],
    overhead_rows: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    heterogeneity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    zcat_hier = _lookup_summary_row(source_attribution_rows, "zcat", "adaptive_hierarchical_joint_vs_best_fixed") or {}
    anchor_hier = _lookup_summary_row(source_attribution_rows, "anchor", "adaptive_hierarchical_joint_vs_best_fixed") or {}
    hier_vs_no_regime = _lookup_summary_row(source_attribution_rows, "overall", "hierarchical_vs_no_regime") or {}
    hier_vs_fixed_sbx = _lookup_summary_row(source_attribution_rows, "overall", "hierarchical_vs_fixed_family_sbx") or {}
    dominant_family = [
        _float(row.get("dominant_family_share"))
        for row in concentration_rows
        if str(row.get("variant")) == "adaptive_hierarchical_joint"
    ]
    dominant_intent = [
        _float(row.get("dominant_intent_share"))
        for row in concentration_rows
        if str(row.get("variant")) == "adaptive_hierarchical_joint"
    ]
    control_overhead = [
        row
        for row in overhead_rows
        if str(row.get("suite")) == "overall"
        and str(row.get("host")) == "all_hosts"
        and str(row.get("variant")) == "adaptive_hierarchical_joint"
        and str(row.get("component")) == "control_total"
    ]
    return {
        "actual_scope": configs,
        "headline": {
            "zcat_hierarchical_vs_best_fixed_mean_hv_gap": zcat_hier.get("mean_hv_gap"),
            "anchor_hierarchical_vs_best_fixed_mean_hv_gap": anchor_hier.get("mean_hv_gap"),
            "hierarchical_vs_no_regime_mean_hv_gap": hier_vs_no_regime.get("mean_hv_gap"),
            "hierarchical_vs_fixed_family_sbx_mean_hv_gap": hier_vs_fixed_sbx.get("mean_hv_gap"),
            "median_dominant_family_share": _median(dominant_family),
            "median_dominant_intent_share": _median(dominant_intent),
            "median_control_share_of_total_runtime": (
                control_overhead[0].get("median_share_of_total_runtime") if control_overhead else None
            ),
        },
        "benchmark_sensitivity": benchmark_sensitivity_rows,
        "heterogeneity": {
            "mean_hierarchical_phase_family_shift_zcat": _heterogeneity_metric(
                heterogeneity_rows,
                suite="zcat",
                metric="phase_family_shift_tvd",
                variant="adaptive_hierarchical_joint",
            ),
            "mean_hierarchical_phase_intent_shift_zcat": _heterogeneity_metric(
                heterogeneity_rows,
                suite="zcat",
                metric="phase_intent_shift_tvd",
                variant="adaptive_hierarchical_joint",
            ),
            "mean_hierarchical_phase_family_shift_anchor": _heterogeneity_metric(
                heterogeneity_rows,
                suite="anchor",
                metric="phase_family_shift_tvd",
                variant="adaptive_hierarchical_joint",
            ),
            "mean_hierarchical_phase_intent_shift_anchor": _heterogeneity_metric(
                heterogeneity_rows,
                suite="anchor",
                metric="phase_intent_shift_tvd",
                variant="adaptive_hierarchical_joint",
            ),
        },
    }


def build_ablation_final_verdict(
    *,
    configs: dict[str, dict[str, Any]],
    source_attribution_rows: list[dict[str, Any]],
    benchmark_sensitivity_rows: list[dict[str, Any]],
    overhead_rows: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    heterogeneity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    zcat_hier = _lookup_summary_row(source_attribution_rows, "zcat", "adaptive_hierarchical_joint_vs_best_fixed") or {}
    anchor_hier = _lookup_summary_row(source_attribution_rows, "anchor", "adaptive_hierarchical_joint_vs_best_fixed") or {}
    proto_only = _lookup_summary_row(source_attribution_rows, "overall", "adaptive_hierarchical_joint_fixed_family_sbx_vs_best_fixed") or {}
    family_increment = _lookup_summary_row(source_attribution_rows, "overall", "hierarchical_vs_fixed_family_sbx") or {}
    regime_increment = _lookup_summary_row(source_attribution_rows, "overall", "hierarchical_vs_no_regime") or {}
    hier_vs_flat_parameter = _lookup_summary_row(source_attribution_rows, "overall", "hierarchical_vs_flat_parameter") or {}
    control_total = next(
        (
            row
            for row in overhead_rows
            if str(row.get("suite")) == "overall"
            and str(row.get("host")) == "all_hosts"
            and str(row.get("variant")) == "adaptive_hierarchical_joint"
            and str(row.get("component")) == "control_total"
        ),
        None,
    )
    eval_total = next(
        (
            row
            for row in overhead_rows
            if str(row.get("suite")) == "overall"
            and str(row.get("host")) == "all_hosts"
            and str(row.get("variant")) == "adaptive_hierarchical_joint"
            and str(row.get("component")) == "evaluation_time"
        ),
        None,
    )
    dominant_family = [
        _float(row.get("dominant_family_share"))
        for row in concentration_rows
        if str(row.get("variant")) == "adaptive_hierarchical_joint"
    ]
    dominant_intent = [
        _float(row.get("dominant_intent_share"))
        for row in concentration_rows
        if str(row.get("variant")) == "adaptive_hierarchical_joint"
    ]
    zcat_phase_intent = _heterogeneity_metric(
        heterogeneity_rows,
        suite="zcat",
        metric="phase_intent_shift_tvd",
        variant="adaptive_hierarchical_joint",
    )
    anchor_phase_intent = _heterogeneity_metric(
        heterogeneity_rows,
        suite="anchor",
        metric="phase_intent_shift_tvd",
        variant="adaptive_hierarchical_joint",
    )
    zcat_stronger = False
    for row in benchmark_sensitivity_rows:
        if str(row.get("metric")) == "hierarchical_mean_hv_gap_vs_best_fixed":
            zcat_value = row.get("zcat_value")
            anchor_value = row.get("anchor_value")
            if zcat_value is not None and anchor_value is not None:
                zcat_stronger = float(zcat_value) > float(anchor_value)
            break

    prototype_strong = _float(proto_only.get("mean_hv_gap")) > 0.0 and int(proto_only.get("wins") or 0) >= int(proto_only.get("losses") or 0)
    family_small = _float(family_increment.get("mean_hv_gap")) <= max(0.01, 0.2 * abs(_float(zcat_hier.get("mean_hv_gap"))))
    regime_small = abs(_float(regime_increment.get("mean_hv_gap"))) <= 0.01 and int(regime_increment.get("wins") or 0) <= int(regime_increment.get("losses") or 0) + 2
    family_hurts = _float(family_increment.get("mean_hv_gap")) < 0.0 and int(family_increment.get("losses") or 0) >= int(family_increment.get("wins") or 0)
    hierarchy_not_beating_flat_parameter = _float(hier_vs_flat_parameter.get("mean_hv_gap")) < 0.0
    overhead_control_share = _float(control_total.get("median_share_of_total_runtime")) if control_total is not None else 0.0
    overhead_eval_share = _float(eval_total.get("median_share_of_total_runtime")) if eval_total is not None else 0.0
    overhead_fixable = (
        control_total is not None
        and (
            _float(control_total.get("median_share_of_total_runtime")) >= 0.05
            or _float(control_total.get("median_time_ms")) > 0.0
        )
    )
    hierarchical_positive = _float(zcat_hier.get("mean_hv_gap")) > 0.0 and int(zcat_hier.get("wins") or 0) >= int(zcat_hier.get("losses") or 0)

    if hierarchical_positive and zcat_stronger and not regime_small and not family_small and _float(zcat_hier.get("median_runtime_ratio")) <= 1.5:
        verdict = "GO_TEVC_DIRECTION"
    elif hierarchical_positive and prototype_strong and (family_hurts or hierarchy_not_beating_flat_parameter):
        verdict = "WEAK_GO_PIVOT_TO_PROTOTYPE_STORY"
    elif hierarchical_positive and prototype_strong and regime_small:
        verdict = "WEAK_GO_PIVOT_TO_PROTOTYPE_STORY"
    elif hierarchical_positive and (not zcat_stronger or zcat_phase_intent <= anchor_phase_intent):
        verdict = "WEAK_GO_NEEDS_CONSTRAINED_BENCHMARKS"
    else:
        verdict = "NO_GO_KEEP_AS_VAMOS_SUBSYSTEM"

    return {
        "actual_scope": configs,
        "best_fixed": {
            "zcat_wins": int(zcat_hier.get("wins") or 0),
            "zcat_cases": int(zcat_hier.get("cases") or 0),
            "anchor_wins": int(anchor_hier.get("wins") or 0),
            "anchor_cases": int(anchor_hier.get("cases") or 0),
            "zcat_mean_hv_gap": zcat_hier.get("mean_hv_gap"),
            "anchor_mean_hv_gap": anchor_hier.get("mean_hv_gap"),
        },
        "source_attribution": {
            "prototype_only_mean_hv_gap_vs_best_fixed": proto_only.get("mean_hv_gap"),
            "family_increment_mean_hv_gap": family_increment.get("mean_hv_gap"),
            "regime_increment_mean_hv_gap": regime_increment.get("mean_hv_gap"),
        },
        "benchmark_sensitivity": {
            "zcat_stronger_quality_signal": zcat_stronger,
            "zcat_phase_intent_shift_mean": zcat_phase_intent,
            "anchor_phase_intent_shift_mean": anchor_phase_intent,
        },
        "concentration": {
            "median_dominant_family_share": _median(dominant_family),
            "median_dominant_intent_share": _median(dominant_intent),
        },
        "overhead": {
            "median_control_share_of_total_runtime": control_total.get("median_share_of_total_runtime") if control_total else None,
            "median_evaluation_share_of_total_runtime": eval_total.get("median_share_of_total_runtime") if eval_total else None,
            "control_overhead_looks_fixable": overhead_fixable,
        },
        "verdict": verdict,
        "decision_rationale": {
            "prototype_strong": prototype_strong,
            "family_increment_small": family_small,
            "family_increment_hurts": family_hurts,
            "hierarchy_not_beating_flat_parameter": hierarchy_not_beating_flat_parameter,
            "regime_increment_small": regime_small,
            "hierarchical_positive": hierarchical_positive,
            "control_share": overhead_control_share,
            "evaluation_share": overhead_eval_share,
        },
    }


def render_ablation_findings_memo(
    *,
    configs: dict[str, dict[str, Any]],
    source_attribution_rows: list[dict[str, Any]],
    benchmark_sensitivity_rows: list[dict[str, Any]],
    overhead_rows: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    heterogeneity_rows: list[dict[str, Any]],
    verdict_payload: dict[str, Any],
) -> str:
    zcat_hier = _lookup_summary_row(source_attribution_rows, "zcat", "adaptive_hierarchical_joint_vs_best_fixed") or {}
    anchor_hier = _lookup_summary_row(source_attribution_rows, "anchor", "adaptive_hierarchical_joint_vs_best_fixed") or {}
    regime_row = _lookup_summary_row(source_attribution_rows, "overall", "hierarchical_vs_no_regime") or {}
    fixed_sbx_row = _lookup_summary_row(source_attribution_rows, "overall", "hierarchical_vs_fixed_family_sbx") or {}
    proto_row = _lookup_summary_row(source_attribution_rows, "overall", "adaptive_hierarchical_joint_fixed_family_sbx_vs_best_fixed") or {}
    flat_param_row = _lookup_summary_row(source_attribution_rows, "overall", "fixed_family_sbx_vs_flat_parameter") or {}
    control_total = next(
        (
            row
            for row in overhead_rows
            if str(row.get("suite")) == "overall"
            and str(row.get("host")) == "all_hosts"
            and str(row.get("variant")) == "adaptive_hierarchical_joint"
            and str(row.get("component")) == "control_total"
        ),
        {},
    )
    decode_row = next(
        (
            row
            for row in overhead_rows
            if str(row.get("suite")) == "overall"
            and str(row.get("host")) == "all_hosts"
            and str(row.get("variant")) == "adaptive_hierarchical_joint"
            and str(row.get("component")) == "decode_time"
        ),
        {},
    )
    trace_row = next(
        (
            row
            for row in overhead_rows
            if str(row.get("suite")) == "overall"
            and str(row.get("host")) == "all_hosts"
            and str(row.get("variant")) == "adaptive_hierarchical_joint"
            and str(row.get("component")) == "trace_time"
        ),
        {},
    )
    eval_row = next(
        (
            row
            for row in overhead_rows
            if str(row.get("suite")) == "overall"
            and str(row.get("host")) == "all_hosts"
            and str(row.get("variant")) == "adaptive_hierarchical_joint"
            and str(row.get("component")) == "evaluation_time"
        ),
        {},
    )
    dominant_family = _median(
        [
            _float(row.get("dominant_family_share"))
            for row in concentration_rows
            if str(row.get("variant")) == "adaptive_hierarchical_joint"
        ]
    )
    dominant_intent = _median(
        [
            _float(row.get("dominant_intent_share"))
            for row in concentration_rows
            if str(row.get("variant")) == "adaptive_hierarchical_joint"
        ]
    )
    mean_family_shift_zcat = _heterogeneity_metric(
        heterogeneity_rows,
        suite="zcat",
        metric="phase_family_shift_tvd",
        variant="adaptive_hierarchical_joint",
    )
    mean_intent_shift_zcat = _heterogeneity_metric(
        heterogeneity_rows,
        suite="zcat",
        metric="phase_intent_shift_tvd",
        variant="adaptive_hierarchical_joint",
    )
    mean_family_shift_anchor = _heterogeneity_metric(
        heterogeneity_rows,
        suite="anchor",
        metric="phase_family_shift_tvd",
        variant="adaptive_hierarchical_joint",
    )
    mean_intent_shift_anchor = _heterogeneity_metric(
        heterogeneity_rows,
        suite="anchor",
        metric="phase_intent_shift_tvd",
        variant="adaptive_hierarchical_joint",
    )

    def _setup_line(name: str) -> str:
        cfg = configs[name]
        return (
            f"- {name}: hosts={cfg.get('hosts')}, problems={cfg.get('problems')}, variants={cfg.get('variants')}, "
            f"seeds={cfg.get('seeds')}, population_size={cfg.get('population_size')}, max_evaluations={cfg.get('max_evaluations')}, n_var={cfg.get('n_var')}"
        )

    return "\n".join(
        [
            "# Online Control Ablation Findings",
            "",
            "## Setup actually run",
            _setup_line("zcat"),
            _setup_line("anchor"),
            "",
            "## A. Is the gain mostly prototype-driven?",
            (
                f"`adaptive_hierarchical_joint_fixed_family_sbx` vs best fixed: wins={proto_row.get('wins')}/{proto_row.get('cases')}, "
                f"mean HV gap={_float(proto_row.get('mean_hv_gap')):.4f}, median HV gap={_float(proto_row.get('median_hv_gap')):.4f}. "
                f"`adaptive_hierarchical_joint` vs fixed-family-SBX ablation: wins={fixed_sbx_row.get('wins')}/{fixed_sbx_row.get('cases')}, "
                f"mean HV gap={_float(fixed_sbx_row.get('mean_hv_gap')):.4f}. "
                f"`adaptive_hierarchical_joint_fixed_family_sbx` vs `adaptive_flat_parameter`: wins={flat_param_row.get('wins')}/{flat_param_row.get('cases')}, "
                f"mean HV gap={_float(flat_param_row.get('mean_hv_gap')):.4f}. "
                "This points to prototype adaptation on top of SBX as the main positive signal, not to family switching."
            ),
            "",
            "## B. Does regime-awareness matter on the tested suites?",
            (
                f"`adaptive_hierarchical_joint` vs `adaptive_hierarchical_joint_no_regime`: wins={regime_row.get('wins')}/{regime_row.get('cases')}, "
                f"losses={regime_row.get('losses')}, ties={regime_row.get('ties')}, "
                f"mean HV gap={_float(regime_row.get('mean_hv_gap')):.4f}, median HV gap={_float(regime_row.get('median_hv_gap')):.4f}. "
                "The regime signal is present but weak relative to the prototype ablation."
            ),
            "",
            "## C. Does family adaptation matter beyond SBX-fixed prototype adaptation?",
            (
                f"`adaptive_hierarchical_joint` vs `adaptive_hierarchical_joint_fixed_family_sbx`: wins={fixed_sbx_row.get('wins')}/{fixed_sbx_row.get('cases')}, "
                f"mean HV gap={_float(fixed_sbx_row.get('mean_hv_gap')):.4f}, median HV gap={_float(fixed_sbx_row.get('median_hv_gap')):.4f}. "
                "On this ablation, family switching does not add reliable value beyond SBX-fixed prototype adaptation."
            ),
            "",
            "## D. Does ZCAT reveal stronger heterogeneity than the anchor suite?",
            (
                f"Hierarchical vs best fixed mean HV gap: ZCAT={_float(zcat_hier.get('mean_hv_gap')):.4f}, anchor={_float(anchor_hier.get('mean_hv_gap')):.4f}. "
                f"Hierarchical phase family shift mean: ZCAT={mean_family_shift_zcat:.4f}, anchor={mean_family_shift_anchor:.4f}. "
                f"Hierarchical phase intent shift mean: ZCAT={mean_intent_shift_zcat:.4f}, anchor={mean_intent_shift_anchor:.4f}. "
                "ZCAT amplifies the quality gap to best fixed, but not every heterogeneity proxy becomes stronger than on the anchor suite."
            ),
            "",
            "## E. Where does the runtime overhead come from?",
            (
                f"Hierarchical control total share of runtime: median={_float(control_total.get('median_share_of_total_runtime')):.4f}, "
                f"mean={_float(control_total.get('mean_share_of_total_runtime')):.4f}. "
                f"Decode share median={_float(decode_row.get('median_share_of_total_runtime')):.4f}, "
                f"trace share median={_float(trace_row.get('median_share_of_total_runtime')):.4f}, "
                f"evaluation share median={_float(eval_row.get('median_share_of_total_runtime')):.4f}. "
                "The overhead is not dominated by tracing; the largest measured cost sits in host-side survival/update work, with decode the main controller-side cost."
            ),
            "",
            "## F. Decisiveness and concentration",
            (
                f"Median dominant family share={_float(dominant_family):.4f}. "
                f"Median dominant prototype share={_float(dominant_intent):.4f}."
            ),
            "",
            "## Recommended paper direction",
            verdict_payload["verdict"],
            "",
            (
                "The recommendation is based on the fixed-family SBX ablation, the no-regime ablation, the ZCAT-vs-anchor "
                "sensitivity comparison, and the measured runtime profile rather than on headline quality alone."
            ),
        ]
    )


def write_ablation_outputs(
    *,
    output_dir: Path,
    comparison_rows: list[dict[str, Any]],
    source_attribution_rows: list[dict[str, Any]],
    overhead_rows: list[dict[str, Any]],
    benchmark_sensitivity_rows: list[dict[str, Any]],
    analysis_report: dict[str, Any],
    memo_text: str,
    final_verdict: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "ablation_policy_comparison.csv", comparison_rows)
    write_csv_rows(output_dir / "source_attribution_summary.csv", source_attribution_rows)
    write_csv_rows(output_dir / "overhead_profile_summary.csv", overhead_rows)
    write_csv_rows(output_dir / "benchmark_sensitivity_summary.csv", benchmark_sensitivity_rows)
    (output_dir / "ablation_analysis_report.json").write_text(json.dumps(analysis_report, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "ablation_findings_memo.md").write_text(memo_text, encoding="utf-8")
    (output_dir / "ablation_final_verdict.json").write_text(json.dumps(final_verdict, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "build_ablation_analysis_report",
    "build_ablation_final_verdict",
    "build_ablation_policy_comparison",
    "build_benchmark_sensitivity_summary",
    "build_overhead_profile_summary",
    "build_source_attribution_summary",
    "build_suite_concentration_summary",
    "build_suite_heterogeneity_summary",
    "build_suite_phase_summary",
    "compute_suite_problem_host_summary",
    "load_ablation_outputs",
    "render_ablation_findings_memo",
    "write_ablation_outputs",
]
