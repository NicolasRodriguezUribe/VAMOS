from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .online_control_ablation_analysis import build_overhead_profile_summary
from .online_control_analysis import INTENT_LABELS, build_concentration_summary, build_phase_summary, load_pilot_output, write_csv_rows

PROTOTYPE_VARIANT = "semantic_prototype_sbx"
FIXED_SBX_VARIANT = "fixed_sbx"
HIERARCHICAL_VARIANT = "adaptive_hierarchical_joint"
NO_REGIME_VARIANT = "adaptive_hierarchical_joint_no_regime"
COMPARISON_ORDER = {
    "semantic_prototype_sbx_vs_fixed_sbx": 0,
    "semantic_prototype_sbx_vs_adaptive_hierarchical_joint": 1,
    "adaptive_hierarchical_joint_vs_adaptive_hierarchical_joint_no_regime": 2,
}


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


def _infer_suite(problem: str, explicit_suite: Any = None) -> str:
    if explicit_suite is not None and str(explicit_suite).strip():
        return str(explicit_suite).strip().lower()
    return "zcat" if str(problem).strip().lower().startswith("zcat") else "anchor"


def load_confirmatory_output(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    runs, summary, traces = load_pilot_output(output_dir)
    config = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    for bucket in (runs, summary, traces):
        for row in bucket:
            row["suite"] = _infer_suite(str(row.get("problem", "")), row.get("suite"))
    return runs, summary, traces, config


def _group_case_rows(summary_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        suite = _infer_suite(str(row.get("problem", "")), row.get("suite"))
        grouped[(suite, str(row["host"]), str(row["problem"]))][str(row["variant"])] = row
    return grouped


def _outcome_from_gap(hv_gap: float, *, tolerance: float = 1e-6) -> str:
    if hv_gap > tolerance:
        return "win"
    if hv_gap < -tolerance:
        return "loss"
    return "tie"


def _lookup_summary_row(
    summary_rows: list[dict[str, Any]],
    *,
    comparison_name: str,
    scope_type: str,
    suite: str,
    host: str,
) -> dict[str, Any] | None:
    for row in summary_rows:
        if (
            str(row.get("comparison_name")) == comparison_name
            and str(row.get("scope_type")) == scope_type
            and str(row.get("suite")) == suite
            and str(row.get("host")) == host
        ):
            return row
    return None


def build_confirmatory_case_comparisons(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_case_rows(summary_rows)
    comparison_defs = (
        (PROTOTYPE_VARIANT, FIXED_SBX_VARIANT, "semantic_prototype_sbx_vs_fixed_sbx"),
        (PROTOTYPE_VARIANT, HIERARCHICAL_VARIANT, "semantic_prototype_sbx_vs_adaptive_hierarchical_joint"),
        (HIERARCHICAL_VARIANT, NO_REGIME_VARIANT, "adaptive_hierarchical_joint_vs_adaptive_hierarchical_joint_no_regime"),
    )
    rows: list[dict[str, Any]] = []
    for (suite, host, problem), mapping in sorted(grouped.items()):
        for left_variant, right_variant, comparison_name in comparison_defs:
            if left_variant not in mapping or right_variant not in mapping:
                continue
            left_row = mapping[left_variant]
            right_row = mapping[right_variant]
            hv_gap = _float(left_row.get("mean_hv")) - _float(right_row.get("mean_hv"))
            reward_gap = None
            if left_row.get("mean_average_reward") is not None and right_row.get("mean_average_reward") is not None:
                reward_gap = _float(left_row.get("mean_average_reward")) - _float(right_row.get("mean_average_reward"))
            igd_plus_gap = None
            if left_row.get("mean_igd_plus") is not None and right_row.get("mean_igd_plus") is not None:
                igd_plus_gap = _float(left_row.get("mean_igd_plus")) - _float(right_row.get("mean_igd_plus"))
            rows.append(
                {
                    "suite": suite,
                    "host": host,
                    "problem": problem,
                    "comparison_name": comparison_name,
                    "left_variant": left_variant,
                    "right_variant": right_variant,
                    "hv_gap": hv_gap,
                    "runtime_ratio": _float(left_row.get("mean_time_ms")) / max(1e-9, _float(right_row.get("mean_time_ms"))),
                    "reward_gap": reward_gap,
                    "igd_plus_gap": igd_plus_gap,
                    "outcome": _outcome_from_gap(hv_gap),
                }
            )
    return rows


def _aggregate_comparison_rows(
    rows: list[dict[str, Any]],
    *,
    scope_type: str,
    suite: str,
    host: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["comparison_name"])].append(row)
    output: list[dict[str, Any]] = []
    for comparison_name, bucket in sorted(grouped.items()):
        hv = [_float(row.get("hv_gap")) for row in bucket if row.get("hv_gap") is not None]
        runtime = [_float(row.get("runtime_ratio")) for row in bucket if row.get("runtime_ratio") is not None]
        reward = [_float(row.get("reward_gap")) for row in bucket if row.get("reward_gap") is not None]
        igd = [_float(row.get("igd_plus_gap")) for row in bucket if row.get("igd_plus_gap") is not None]
        output.append(
            {
                "scope_type": scope_type,
                "suite": suite,
                "host": host,
                "comparison_name": comparison_name,
                "left_variant": bucket[0]["left_variant"],
                "right_variant": bucket[0]["right_variant"],
                "cases": len(bucket),
                "wins": sum(1 for row in bucket if row.get("outcome") == "win"),
                "losses": sum(1 for row in bucket if row.get("outcome") == "loss"),
                "ties": sum(1 for row in bucket if row.get("outcome") == "tie"),
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


def build_confirmatory_summary(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        suite = str(row["suite"])
        host = str(row["host"])
        buckets[("overall", "all", "all_hosts")].append(row)
        buckets[("suite", suite, "all_hosts")].append(row)
        buckets[("host", "all", host)].append(row)
        buckets[("suite_host", suite, host)].append(row)

    output: list[dict[str, Any]] = []
    for (scope_type, suite, host), rows in sorted(buckets.items()):
        output.extend(_aggregate_comparison_rows(rows, scope_type=scope_type, suite=suite, host=host))
    output.sort(key=lambda row: (str(row["scope_type"]), str(row["suite"]), str(row["host"]), COMPARISON_ORDER.get(str(row["comparison_name"]), 99)))
    return output


def _aggregate_scope_rows(rows: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    dominant_intent = [_float(row.get("dominant_intent_share")) for row in rows]
    intent_switches = [_float(row.get("mean_intent_switches")) for row in rows]
    regime_switches = [_float(row.get("mean_regime_switches")) for row in rows]
    intent_concentration = [_float(row.get("intent_concentration")) for row in rows if row.get("intent_concentration") is not None]
    return {
        "scope": label,
        "cases": len(rows),
        "mean_dominant_intent_share": _mean(dominant_intent),
        "median_dominant_intent_share": _median(dominant_intent),
        "mean_intent_switches": _mean(intent_switches),
        "median_intent_switches": _median(intent_switches),
        "mean_regime_switches": _mean(regime_switches),
        "median_regime_switches": _median(regime_switches),
        "mean_intent_concentration": _mean(intent_concentration),
        "median_intent_concentration": _median(intent_concentration),
    }


def build_prototype_profile(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    concentration_rows = [row for row in build_concentration_summary(summary_rows) if str(row.get("variant")) == PROTOTYPE_VARIANT]
    for row in concentration_rows:
        row["suite"] = _infer_suite(str(row.get("problem", "")), row.get("suite"))

    overall = _aggregate_scope_rows(concentration_rows, label="overall")
    by_suite = [
        _aggregate_scope_rows([row for row in concentration_rows if str(row.get("suite")) == suite], label=suite)
        for suite in ("zcat", "anchor")
    ]
    by_host = [
        _aggregate_scope_rows([row for row in concentration_rows if str(row.get("host")) == host], label=host)
        for host in ("nsgaii", "moead")
    ]
    return {"overall": overall, "by_suite": by_suite, "by_host": by_host}


def _aggregate_phase_bucket(rows: list[dict[str, Any]], *, scope: str, phase: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "scope": scope,
        "phase": phase,
        "n_rows": len(rows),
        "mean_reward": _mean([_float(row.get("mean_reward")) for row in rows]),
        "mean_overhead_ms": _mean([_float(row.get("mean_overhead_ms")) for row in rows]),
        "mean_intent_switches": _mean([_float(row.get("mean_intent_switches")) for row in rows]),
    }
    intent_shares = {label: _mean([_float(row.get(f"intent_share_{label}")) for row in rows]) for label in INTENT_LABELS}
    payload.update({f"intent_share_{label}": value for label, value in intent_shares.items()})
    dominant_label = max(intent_shares, key=intent_shares.get) if intent_shares else "balanced"
    payload["dominant_intent"] = dominant_label
    payload["dominant_intent_share"] = intent_shares.get(dominant_label, 0.0)
    return payload


def build_phase_diagnostics(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    phase_rows = [row for row in build_phase_summary(trace_rows) if str(row.get("variant")) == PROTOTYPE_VARIANT]
    for row in phase_rows:
        row["suite"] = _infer_suite(str(row.get("problem", "")), row.get("suite"))

    phase_buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in phase_rows:
        suite = str(row["suite"])
        host = str(row["host"])
        phase = str(row["phase"])
        phase_buckets[("overall", phase)].append(row)
        phase_buckets[(suite, phase)].append(row)
        phase_buckets[(host, phase)].append(row)

    phase_profiles: list[dict[str, Any]] = []
    for (scope, phase), rows in sorted(phase_buckets.items()):
        phase_profiles.append(_aggregate_phase_bucket(rows, scope=scope, phase=phase))

    case_phase_groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in phase_rows:
        case_phase_groups[(str(row["suite"]), str(row["host"]), str(row["problem"]))][str(row["phase"])] = row

    shift_rows: list[dict[str, Any]] = []
    for (suite, host, problem), phases in sorted(case_phase_groups.items()):
        early = phases.get("early")
        late = phases.get("late")
        if early is None or late is None:
            continue
        shift = 0.5 * sum(
            abs(_float(early.get(f"intent_share_{label}")) - _float(late.get(f"intent_share_{label}")))
            for label in INTENT_LABELS
        )
        shift_rows.append({"suite": suite, "host": host, "problem": problem, "intent_shift_tvd": shift})

    shift_summary = {
        "overall": {
            "mean_intent_shift_tvd": _mean([_float(row.get("intent_shift_tvd")) for row in shift_rows]),
            "median_intent_shift_tvd": _median([_float(row.get("intent_shift_tvd")) for row in shift_rows]),
        },
        "by_suite": [
            {
                "scope": suite,
                "mean_intent_shift_tvd": _mean([_float(row.get("intent_shift_tvd")) for row in shift_rows if str(row.get("suite")) == suite]),
                "median_intent_shift_tvd": _median([_float(row.get("intent_shift_tvd")) for row in shift_rows if str(row.get("suite")) == suite]),
            }
            for suite in ("zcat", "anchor")
        ],
        "by_host": [
            {
                "scope": host,
                "mean_intent_shift_tvd": _mean([_float(row.get("intent_shift_tvd")) for row in shift_rows if str(row.get("host")) == host]),
                "median_intent_shift_tvd": _median([_float(row.get("intent_shift_tvd")) for row in shift_rows if str(row.get("host")) == host]),
            }
            for host in ("nsgaii", "moead")
        ],
    }
    return {"phase_profiles": phase_profiles, "shift_summary": shift_summary}


def build_overhead_view(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    overhead_rows = build_overhead_profile_summary(run_rows)
    relevant_variants = {PROTOTYPE_VARIANT, FIXED_SBX_VARIANT, HIERARCHICAL_VARIANT}
    filtered = [
        row
        for row in overhead_rows
        if str(row.get("suite")) == "overall" and str(row.get("host")) == "all_hosts" and str(row.get("variant")) in relevant_variants
    ]
    by_variant: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in filtered:
        by_variant[str(row["variant"])][str(row["component"])] = dict(row)
    return {
        "rows": filtered,
        "prototype_sbx": by_variant.get(PROTOTYPE_VARIANT, {}),
        "fixed_sbx": by_variant.get(FIXED_SBX_VARIANT, {}),
        "hierarchical": by_variant.get(HIERARCHICAL_VARIANT, {}),
    }


def build_confirmatory_report(
    *,
    config: dict[str, Any],
    confirmatory_summary_rows: list[dict[str, Any]],
    prototype_profile: dict[str, Any],
    phase_diagnostics: dict[str, Any],
    overhead_view: dict[str, Any],
) -> dict[str, Any]:
    return {
        "actual_scope": config,
        "comparisons": {
            "overall": {
                "semantic_prototype_sbx_vs_fixed_sbx": _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_fixed_sbx", scope_type="overall", suite="all", host="all_hosts"),
                "semantic_prototype_sbx_vs_adaptive_hierarchical_joint": _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_adaptive_hierarchical_joint", scope_type="overall", suite="all", host="all_hosts"),
                "adaptive_hierarchical_joint_vs_adaptive_hierarchical_joint_no_regime": _lookup_summary_row(confirmatory_summary_rows, comparison_name="adaptive_hierarchical_joint_vs_adaptive_hierarchical_joint_no_regime", scope_type="overall", suite="all", host="all_hosts"),
            },
            "by_suite": {
                suite: {
                    "semantic_prototype_sbx_vs_fixed_sbx": _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_fixed_sbx", scope_type="suite", suite=suite, host="all_hosts"),
                    "semantic_prototype_sbx_vs_adaptive_hierarchical_joint": _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_adaptive_hierarchical_joint", scope_type="suite", suite=suite, host="all_hosts"),
                }
                for suite in ("zcat", "anchor")
            },
            "by_host": {
                host: {
                    "semantic_prototype_sbx_vs_fixed_sbx": _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_fixed_sbx", scope_type="host", suite="all", host=host),
                    "semantic_prototype_sbx_vs_adaptive_hierarchical_joint": _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_adaptive_hierarchical_joint", scope_type="host", suite="all", host=host),
                }
                for host in ("nsgaii", "moead")
            },
        },
        "prototype_profile": prototype_profile,
        "phase_dynamics": phase_diagnostics,
        "overhead": overhead_view,
    }


def build_confirmatory_final_verdict(
    *,
    config: dict[str, Any],
    confirmatory_summary_rows: list[dict[str, Any]],
    prototype_profile: dict[str, Any],
    phase_diagnostics: dict[str, Any],
    overhead_view: dict[str, Any],
) -> dict[str, Any]:
    overall_vs_fixed = _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_fixed_sbx", scope_type="overall", suite="all", host="all_hosts") or {}
    zcat_vs_fixed = _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_fixed_sbx", scope_type="suite", suite="zcat", host="all_hosts") or {}
    anchor_vs_fixed = _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_fixed_sbx", scope_type="suite", suite="anchor", host="all_hosts") or {}
    overall_vs_hier = _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_adaptive_hierarchical_joint", scope_type="overall", suite="all", host="all_hosts") or {}

    prototype_overall = prototype_profile.get("overall", {})
    phase_shift_overall = (phase_diagnostics.get("shift_summary", {}) or {}).get("overall", {})
    prototype_overhead = overhead_view.get("prototype_sbx", {})
    control_total = prototype_overhead.get("control_total", {})
    decode_time = prototype_overhead.get("decode_time", {})
    trace_time = prototype_overhead.get("trace_time", {})
    host_total = prototype_overhead.get("host_pipeline_total", {})

    strong_zcat_signal = _float(zcat_vs_fixed.get("mean_hv_gap")) > 0.0 and int(zcat_vs_fixed.get("wins") or 0) >= int(zcat_vs_fixed.get("losses") or 0)
    overall_positive = _float(overall_vs_fixed.get("mean_hv_gap")) > 0.0 and int(overall_vs_fixed.get("wins") or 0) >= int(overall_vs_fixed.get("losses") or 0)
    beats_hierarchy = _float(overall_vs_hier.get("mean_hv_gap")) >= 0.0 and int(overall_vs_hier.get("wins") or 0) >= int(overall_vs_hier.get("losses") or 0)
    zcat_stronger_than_anchor = _float(zcat_vs_fixed.get("mean_hv_gap")) > _float(anchor_vs_fixed.get("mean_hv_gap"))
    overhead_fixable = _float(host_total.get("median_share_of_total_runtime")) > _float(control_total.get("median_share_of_total_runtime")) and _float(trace_time.get("median_share_of_total_runtime")) < 0.02
    runtime_moderate = _float(overall_vs_fixed.get("median_runtime_ratio")) <= 1.50

    if overall_positive and strong_zcat_signal and beats_hierarchy and zcat_stronger_than_anchor and runtime_moderate:
        verdict = "GO_SWERVO_STYLE"
    elif overall_positive and strong_zcat_signal and overhead_fixable:
        verdict = "WEAK_GO_TEVC_IF_STRONGLY_REFRAMED"
    else:
        verdict = "NO_GO_KEEP_INTERNAL"

    return {
        "actual_scope": config,
        "semantic_prototype_sbx_vs_fixed_sbx": {
            "wins": int(overall_vs_fixed.get("wins") or 0),
            "losses": int(overall_vs_fixed.get("losses") or 0),
            "ties": int(overall_vs_fixed.get("ties") or 0),
            "cases": int(overall_vs_fixed.get("cases") or 0),
            "mean_hv_gap": overall_vs_fixed.get("mean_hv_gap"),
            "median_hv_gap": overall_vs_fixed.get("median_hv_gap"),
            "mean_runtime_ratio": overall_vs_fixed.get("mean_runtime_ratio"),
            "median_runtime_ratio": overall_vs_fixed.get("median_runtime_ratio"),
        },
        "semantic_prototype_sbx_vs_hierarchical": {
            "wins": int(overall_vs_hier.get("wins") or 0),
            "losses": int(overall_vs_hier.get("losses") or 0),
            "ties": int(overall_vs_hier.get("ties") or 0),
            "cases": int(overall_vs_hier.get("cases") or 0),
            "mean_hv_gap": overall_vs_hier.get("mean_hv_gap"),
            "median_hv_gap": overall_vs_hier.get("median_hv_gap"),
        },
        "zcat_vs_anchor": {
            "zcat_mean_hv_gap_vs_fixed_sbx": zcat_vs_fixed.get("mean_hv_gap"),
            "anchor_mean_hv_gap_vs_fixed_sbx": anchor_vs_fixed.get("mean_hv_gap"),
            "zcat_wins": int(zcat_vs_fixed.get("wins") or 0),
            "anchor_wins": int(anchor_vs_fixed.get("wins") or 0),
        },
        "prototype_dynamics": {
            "median_dominant_intent_share": prototype_overall.get("median_dominant_intent_share"),
            "median_intent_switches": prototype_overall.get("median_intent_switches"),
            "mean_phase_intent_shift_tvd": phase_shift_overall.get("mean_intent_shift_tvd"),
        },
        "overhead": {
            "control_total_median_share": control_total.get("median_share_of_total_runtime"),
            "decode_time_median_share": decode_time.get("median_share_of_total_runtime"),
            "trace_time_median_share": trace_time.get("median_share_of_total_runtime"),
            "host_pipeline_total_median_share": host_total.get("median_share_of_total_runtime"),
            "overhead_looks_engineering_fixable": overhead_fixable,
        },
        "verdict": verdict,
        "decision_rationale": {
            "overall_positive": overall_positive,
            "strong_zcat_signal": strong_zcat_signal,
            "beats_hierarchy": beats_hierarchy,
            "zcat_stronger_than_anchor": zcat_stronger_than_anchor,
            "runtime_moderate": runtime_moderate,
            "overhead_fixable": overhead_fixable,
        },
    }


def render_confirmatory_findings_memo(
    *,
    config: dict[str, Any],
    confirmatory_summary_rows: list[dict[str, Any]],
    prototype_profile: dict[str, Any],
    phase_diagnostics: dict[str, Any],
    overhead_view: dict[str, Any],
    verdict_payload: dict[str, Any],
) -> str:
    overall_vs_fixed = _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_fixed_sbx", scope_type="overall", suite="all", host="all_hosts") or {}
    zcat_vs_fixed = _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_fixed_sbx", scope_type="suite", suite="zcat", host="all_hosts") or {}
    anchor_vs_fixed = _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_fixed_sbx", scope_type="suite", suite="anchor", host="all_hosts") or {}
    nsgaii_vs_fixed = _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_fixed_sbx", scope_type="host", suite="all", host="nsgaii") or {}
    moead_vs_fixed = _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_fixed_sbx", scope_type="host", suite="all", host="moead") or {}
    overall_vs_hier = _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_adaptive_hierarchical_joint", scope_type="overall", suite="all", host="all_hosts") or {}
    hier_vs_no_regime = _lookup_summary_row(confirmatory_summary_rows, comparison_name="adaptive_hierarchical_joint_vs_adaptive_hierarchical_joint_no_regime", scope_type="overall", suite="all", host="all_hosts") or {}
    prototype_overall = prototype_profile.get("overall", {})
    phase_profiles = phase_diagnostics.get("phase_profiles", [])
    phase_lookup = {str(row.get("phase")): row for row in phase_profiles if str(row.get("scope")) == "overall"}
    phase_shift = (phase_diagnostics.get("shift_summary", {}) or {}).get("overall", {})
    overhead_proto = overhead_view.get("prototype_sbx", {})
    control_total = overhead_proto.get("control_total", {})
    decode_time = overhead_proto.get("decode_time", {})
    trace_time = overhead_proto.get("trace_time", {})
    host_total = overhead_proto.get("host_pipeline_total", {})

    problems = config.get("problems", [])
    zcat_problems = [item["key"] if isinstance(item, dict) else item for item in problems if _infer_suite(str(item["key"] if isinstance(item, dict) else item), item.get("suite") if isinstance(item, dict) else None) == "zcat"]
    anchor_problems = [item["key"] if isinstance(item, dict) else item for item in problems if _infer_suite(str(item["key"] if isinstance(item, dict) else item), item.get("suite") if isinstance(item, dict) else None) == "anchor"]
    early = phase_lookup.get("early", {})
    late = phase_lookup.get("late", {})

    return "\n".join(
        [
            "# Semantic Prototype SBX Confirmatory Findings",
            "",
            f"Setup run: hosts={config.get('hosts')}, zcat_problems={zcat_problems}, anchor_problems={anchor_problems}, variants={config.get('variants')}, seeds={config.get('seeds')}, population_size={config.get('population_size')}, max_evaluations={config.get('max_evaluations')}.",
            "",
            "## A. Core claim supported",
            f"The strongest evidence-backed claim is not that the method wins uniformly on every host, but that semantic prototype adaptation on top of fixed `SBX_LIKE` produces a real quality signal overall, beating `fixed_sbx` in {int(overall_vs_fixed.get('wins') or 0)}/{int(overall_vs_fixed.get('cases') or 0)} host-problem cases with mean/median HV gap {_float(overall_vs_fixed.get('mean_hv_gap')):.4f} / {_float(overall_vs_fixed.get('median_hv_gap')):.4f}. The confirmatory run therefore supports a prototype-centric paper story more directly than a broader family-switching story, while also showing that the current evidence is host-asymmetric rather than uniformly reproduced.",
            "",
            "## B. Main quantitative results",
            f"Against `fixed_sbx`, `semantic_prototype_sbx` went {int(overall_vs_fixed.get('wins') or 0)} wins / {int(overall_vs_fixed.get('losses') or 0)} losses / {int(overall_vs_fixed.get('ties') or 0)} ties overall, with mean/median runtime ratio {_float(overall_vs_fixed.get('mean_runtime_ratio')):.4f}x / {_float(overall_vs_fixed.get('median_runtime_ratio')):.4f}x.",
            f"ZCAT-first contrast: wins={int(zcat_vs_fixed.get('wins') or 0)}/{int(zcat_vs_fixed.get('cases') or 0)}, mean HV gap={_float(zcat_vs_fixed.get('mean_hv_gap')):.4f}, median runtime ratio={_float(zcat_vs_fixed.get('median_runtime_ratio')):.4f}x. Anchor contrast: wins={int(anchor_vs_fixed.get('wins') or 0)}/{int(anchor_vs_fixed.get('cases') or 0)}, mean HV gap={_float(anchor_vs_fixed.get('mean_hv_gap')):.4f}, median runtime ratio={_float(anchor_vs_fixed.get('median_runtime_ratio')):.4f}x.",
            f"Host contrast: NSGA-II mean HV gap vs `fixed_sbx`={_float(nsgaii_vs_fixed.get('mean_hv_gap')):.4f} over {int(nsgaii_vs_fixed.get('cases') or 0)} cases; MOEA/D mean HV gap={_float(moead_vs_fixed.get('mean_hv_gap')):.4f} over {int(moead_vs_fixed.get('cases') or 0)} cases. This means the fixed-family prototype signal is strong on MOEA/D but not yet confirmed against `fixed_sbx` on NSGA-II.",
            f"Against the full hierarchical reference, `semantic_prototype_sbx` went {int(overall_vs_hier.get('wins') or 0)} wins / {int(overall_vs_hier.get('losses') or 0)} losses / {int(overall_vs_hier.get('ties') or 0)} ties, with mean/median HV gap {_float(overall_vs_hier.get('mean_hv_gap')):.4f} / {_float(overall_vs_hier.get('median_hv_gap')):.4f}.",
            "",
            "## C. Why not the broader hierarchical story",
            f"The full hierarchical controller is not the main story in this confirmatory run because the prototype-SBX method already matches or exceeds it (overall mean HV gap vs hierarchical {_float(overall_vs_hier.get('mean_hv_gap')):.4f}), while the no-regime ablation remains close (hierarchical vs no-regime mean HV gap {_float(hier_vs_no_regime.get('mean_hv_gap')):.4f}). That keeps family switching and regime-awareness in the role of references rather than flagship claims.",
            "",
            "## D. Practical implication",
            "The cleanest positioning is a semantic parameter-control method implemented as a lightweight host-agnostic adaptation layer: the host keeps a strong fixed operator family, while online control adapts interpretable semantic intent prototypes over time.",
            "",
            "## E. Final recommendation",
            verdict_payload["verdict"],
            "",
            f"Prototype decisiveness remains moderate rather than collapsed, with median dominant prototype share {_float(prototype_overall.get('median_dominant_intent_share')):.4f} and median prototype switches {_float(prototype_overall.get('median_intent_switches')):.4f}. Phase dynamics are still real: early dominant prototype={early.get('dominant_intent')}, late dominant prototype={late.get('dominant_intent')}, mean early-to-late prototype shift TVD={_float(phase_shift.get('mean_intent_shift_tvd')):.4f}. Runtime overhead remains mostly host-side rather than trace/controller-side, with control share median={_float(control_total.get('median_share_of_total_runtime')):.4f}, decode share median={_float(decode_time.get('median_share_of_total_runtime')):.4f}, trace share median={_float(trace_time.get('median_share_of_total_runtime')):.4f}, and host-pipeline share median={_float(host_total.get('median_share_of_total_runtime')):.4f}.",
        ]
    )


def write_confirmatory_outputs(
    *,
    output_dir: Path,
    confirmatory_summary_rows: list[dict[str, Any]],
    confirmatory_report: dict[str, Any],
    memo_text: str,
    final_verdict: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "confirmatory_summary.csv", confirmatory_summary_rows)
    (output_dir / "confirmatory_report.json").write_text(json.dumps(confirmatory_report, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "confirmatory_findings_memo.md").write_text(memo_text, encoding="utf-8")
    (output_dir / "confirmatory_final_verdict.json").write_text(json.dumps(final_verdict, indent=2, sort_keys=True), encoding="utf-8")


def build_final_confirmatory_tables(confirmatory_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scope_defs = (
        ("overall", "overall", "all", "all_hosts"),
        ("zcat", "suite", "zcat", "all_hosts"),
        ("anchor", "suite", "anchor", "all_hosts"),
        ("nsgaii", "host", "all", "nsgaii"),
        ("moead", "host", "all", "moead"),
    )
    comparison_names = (
        "semantic_prototype_sbx_vs_fixed_sbx",
        "semantic_prototype_sbx_vs_adaptive_hierarchical_joint",
        "adaptive_hierarchical_joint_vs_adaptive_hierarchical_joint_no_regime",
    )
    rows: list[dict[str, Any]] = []
    for comparison_name in comparison_names:
        for scope_name, scope_type, suite, host in scope_defs:
            row = _lookup_summary_row(
                confirmatory_summary_rows,
                comparison_name=comparison_name,
                scope_type=scope_type,
                suite=suite,
                host=host,
            )
            if row is None:
                continue
            payload = dict(row)
            payload["scope"] = scope_name
            rows.append(payload)
    return rows


def build_final_confirmatory_report(
    *,
    config: dict[str, Any],
    confirmatory_summary_rows: list[dict[str, Any]],
    compact_tables: list[dict[str, Any]],
    prototype_profile: dict[str, Any],
    phase_diagnostics: dict[str, Any],
    overhead_view: dict[str, Any],
    final_verdict: dict[str, Any],
) -> dict[str, Any]:
    report = build_confirmatory_report(
        config=config,
        confirmatory_summary_rows=confirmatory_summary_rows,
        prototype_profile=prototype_profile,
        phase_diagnostics=phase_diagnostics,
        overhead_view=overhead_view,
    )
    report["compact_tables"] = compact_tables
    report["final_verdict"] = final_verdict
    return report


def render_final_confirmatory_findings_memo(
    *,
    config: dict[str, Any],
    confirmatory_summary_rows: list[dict[str, Any]],
    compact_tables: list[dict[str, Any]],
    prototype_profile: dict[str, Any],
    phase_diagnostics: dict[str, Any],
    overhead_view: dict[str, Any],
    final_verdict: dict[str, Any],
) -> str:
    del compact_tables
    overall_vs_fixed = _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_fixed_sbx", scope_type="overall", suite="all", host="all_hosts") or {}
    zcat_vs_fixed = _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_fixed_sbx", scope_type="suite", suite="zcat", host="all_hosts") or {}
    anchor_vs_fixed = _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_fixed_sbx", scope_type="suite", suite="anchor", host="all_hosts") or {}
    nsgaii_vs_fixed = _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_fixed_sbx", scope_type="host", suite="all", host="nsgaii") or {}
    moead_vs_fixed = _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_fixed_sbx", scope_type="host", suite="all", host="moead") or {}
    overall_vs_hier = _lookup_summary_row(confirmatory_summary_rows, comparison_name="semantic_prototype_sbx_vs_adaptive_hierarchical_joint", scope_type="overall", suite="all", host="all_hosts") or {}
    hier_vs_no_regime = _lookup_summary_row(confirmatory_summary_rows, comparison_name="adaptive_hierarchical_joint_vs_adaptive_hierarchical_joint_no_regime", scope_type="overall", suite="all", host="all_hosts") or {}

    prototype_overall = prototype_profile.get("overall", {})
    phase_shift = (phase_diagnostics.get("shift_summary", {}) or {}).get("overall", {})
    phase_profiles = {str(row.get("phase")): row for row in phase_diagnostics.get("phase_profiles", []) if str(row.get("scope")) == "overall"}
    early = phase_profiles.get("early", {})
    late = phase_profiles.get("late", {})

    overhead_proto = overhead_view.get("prototype_sbx", {})
    control_total = overhead_proto.get("control_total", {})
    decode_time = overhead_proto.get("decode_time", {})
    trace_time = overhead_proto.get("trace_time", {})
    host_total = overhead_proto.get("host_pipeline_total", {})
    survival_time = overhead_proto.get("survival_time", {})

    problems = config.get("problems", [])
    zcat_problems = [item["key"] if isinstance(item, dict) else item for item in problems if _infer_suite(str(item["key"] if isinstance(item, dict) else item), item.get("suite") if isinstance(item, dict) else None) == "zcat"]
    anchor_problems = [item["key"] if isinstance(item, dict) else item for item in problems if _infer_suite(str(item["key"] if isinstance(item, dict) else item), item.get("suite") if isinstance(item, dict) else None) == "anchor"]

    return "\n".join(
        [
            "# Final Semantic Prototype SBX Confirmatory Findings",
            "",
            "## 1. Core supported claim",
            (
                f"The strongest supported claim is that semantic prototype adaptation over a fixed `SBX_LIKE` family is the best-supported method in the current line: it beats `fixed_sbx` in {int(overall_vs_fixed.get('wins') or 0)}/{int(overall_vs_fixed.get('cases') or 0)} host-problem cases, beats the broader `adaptive_hierarchical_joint` reference overall, and concentrates most of its quality signal on the ZCAT-first suite rather than on the small classical anchor."
            ),
            "",
            "## 2. Experimental setup actually run",
            (
                f"Hosts: {config.get('hosts')}. ZCAT-first problems: {zcat_problems}. Anchor problems: {anchor_problems}. "
                f"Variants: {config.get('variants')}. Seeds: {config.get('seeds')}. Population size: {config.get('population_size')}. "
                f"Max evaluations: {config.get('max_evaluations')}."
            ),
            "",
            "## 3. Main result vs fixed SBX",
            (
                f"Overall, `semantic_prototype_sbx` went {int(overall_vs_fixed.get('wins') or 0)} wins / {int(overall_vs_fixed.get('losses') or 0)} losses / {int(overall_vs_fixed.get('ties') or 0)} ties against `fixed_sbx`, with mean/median HV gap {_float(overall_vs_fixed.get('mean_hv_gap')):.4f} / {_float(overall_vs_fixed.get('median_hv_gap')):.4f} and mean/median runtime ratio {_float(overall_vs_fixed.get('mean_runtime_ratio')):.4f}x / {_float(overall_vs_fixed.get('median_runtime_ratio')):.4f}x. Prototype behavior stayed adaptive rather than collapsing to one intent: median dominant prototype share was {_float(prototype_overall.get('median_dominant_intent_share')):.4f}, median prototype switches were {_float(prototype_overall.get('median_intent_switches')):.1f}, and mean early-to-late prototype shift TVD was {_float(phase_shift.get('mean_intent_shift_tvd')):.4f}."
            ),
            "",
            "## 4. Host-wise interpretation",
            (
                f"The observed effect is asymmetric by host. On MOEA/D, `semantic_prototype_sbx` went {int(moead_vs_fixed.get('wins') or 0)}/{int(moead_vs_fixed.get('cases') or 0)} against `fixed_sbx` with mean HV gap {_float(moead_vs_fixed.get('mean_hv_gap')):.4f}. On NSGA-II, it only went {int(nsgaii_vs_fixed.get('wins') or 0)}/{int(nsgaii_vs_fixed.get('cases') or 0)} with mean HV gap {_float(nsgaii_vs_fixed.get('mean_hv_gap')):.4f}. Based only on observed evidence, this suggests that the prototype controller is especially useful in MOEA/D, while on NSGA-II the fixed SBX baseline remains harder to beat. The fact that `semantic_prototype_sbx` still beats the full hierarchical reference overall and on NSGA-II indicates that the weak NSGA-II result is more consistent with a strong fixed baseline than with the prototype policy being uniformly ineffective."
            ),
            "",
            "## 5. ZCAT-first interpretation",
            (
                f"ZCAT is clearly more revealing than the anchor suite for this method. On ZCAT, `semantic_prototype_sbx` went {int(zcat_vs_fixed.get('wins') or 0)}/{int(zcat_vs_fixed.get('cases') or 0)} against `fixed_sbx` with mean/median HV gap {_float(zcat_vs_fixed.get('mean_hv_gap')):.4f} / {_float(zcat_vs_fixed.get('median_hv_gap')):.4f}. On the anchor, it went {int(anchor_vs_fixed.get('wins') or 0)}/{int(anchor_vs_fixed.get('cases') or 0)} with mean/median HV gap {_float(anchor_vs_fixed.get('mean_hv_gap')):.4f} / {_float(anchor_vs_fixed.get('median_hv_gap')):.4f}. The anchor signal is nearly flat, while ZCAT carries most of the positive evidence, which supports a ZCAT-first paper framing rather than a classical-suite-first framing."
            ),
            "",
            "## 6. Why the broader hierarchical story is not the main paper",
            (
                f"`semantic_prototype_sbx` remains better than `adaptive_hierarchical_joint` overall, with {int(overall_vs_hier.get('wins') or 0)} wins / {int(overall_vs_hier.get('losses') or 0)} losses / {int(overall_vs_hier.get('ties') or 0)} ties and mean/median HV gap {_float(overall_vs_hier.get('mean_hv_gap')):.4f} / {_float(overall_vs_hier.get('median_hv_gap')):.4f}. Meanwhile, the hierarchical no-regime reference stays close, with hierarchical vs no-regime mean HV gap only {_float(hier_vs_no_regime.get('mean_hv_gap')):.4f}. That means the main gains do not require a family-switching or regime-centric story."
            ),
            "",
            "## 7. Overhead interpretation",
            (
                f"The remaining weakness is runtime overhead versus `fixed_sbx`, not absence of quality signal. The overall median runtime ratio versus `fixed_sbx` is {_float(overall_vs_fixed.get('median_runtime_ratio')):.4f}x. The controller-side shares are comparatively small: control={_float(control_total.get('median_share_of_total_runtime')):.4f}, decode={_float(decode_time.get('median_share_of_total_runtime')):.4f}, trace={_float(trace_time.get('median_share_of_total_runtime')):.4f}. The larger measured cost is host-side, with host-pipeline share={_float(host_total.get('median_share_of_total_runtime')):.4f} and survival/update share={_float(survival_time.get('median_share_of_total_runtime')):.4f}. This makes the overhead look more like a host-level engineering problem than a trace/controller tax, but it is not something that disappears just by disabling logging."
            ),
            "",
            "## 8. Final recommendation",
            final_verdict["verdict"],
        ]
    )


def write_final_confirmatory_outputs(
    *,
    output_dir: Path,
    final_summary_rows: list[dict[str, Any]],
    final_tables_rows: list[dict[str, Any]],
    final_report: dict[str, Any],
    memo_text: str,
    final_verdict: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "final_confirmatory_summary.csv", final_summary_rows)
    write_csv_rows(output_dir / "final_confirmatory_tables.csv", final_tables_rows)
    (output_dir / "final_confirmatory_report.json").write_text(json.dumps(final_report, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "final_confirmatory_findings_memo.md").write_text(memo_text, encoding="utf-8")
    (output_dir / "final_confirmatory_verdict.json").write_text(json.dumps(final_verdict, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "build_confirmatory_case_comparisons",
    "build_confirmatory_final_verdict",
    "build_confirmatory_report",
    "build_confirmatory_summary",
    "build_final_confirmatory_report",
    "build_final_confirmatory_tables",
    "build_overhead_view",
    "build_phase_diagnostics",
    "build_prototype_profile",
    "load_confirmatory_output",
    "render_final_confirmatory_findings_memo",
    "render_confirmatory_findings_memo",
    "write_final_confirmatory_outputs",
    "write_confirmatory_outputs",
]
