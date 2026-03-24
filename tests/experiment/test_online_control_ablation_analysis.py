from __future__ import annotations

from vamos.experiment.online_control_ablation_analysis import (
    build_ablation_final_verdict,
    build_benchmark_sensitivity_summary,
    build_overhead_profile_summary,
    build_source_attribution_summary,
    build_suite_concentration_summary,
    build_suite_heterogeneity_summary,
    build_suite_phase_summary,
    compute_suite_problem_host_summary,
)


def _summary_row(
    *,
    suite: str,
    problem: str,
    variant: str,
    mean_hv: float,
    mean_time_ms: float,
    family_sbx: float,
    family_de: float,
    intent_exploratory: float,
    intent_balanced: float,
    intent_local: float,
    intent_mutation: float,
    intent_feasible: float = 0.0,
    family_switches: float = 0.0,
    intent_switches: float = 0.0,
    regime_switches: float = 0.0,
    reward: float = 0.0,
) -> dict[str, object]:
    return {
        "suite": suite,
        "host": "nsgaii",
        "problem": problem,
        "variant": variant,
        "variant_group": "fixed" if variant.startswith("fixed_") else "adaptive",
        "mean_hv": mean_hv,
        "mean_igd_plus": 1.0 / max(mean_hv, 0.1),
        "mean_time_ms": mean_time_ms,
        "mean_average_reward": reward,
        "mean_average_overhead_ms": 1.0,
        "mean_family_concentration": family_sbx * family_sbx + family_de * family_de,
        "mean_regime_concentration": 0.9,
        "mean_intent_concentration": sum(
            share * share
            for share in (intent_exploratory, intent_balanced, intent_local, intent_mutation, intent_feasible)
        ),
        "mean_family_switches": family_switches,
        "mean_regime_switches": regime_switches,
        "mean_intent_switches": intent_switches,
        "mean_family_share_sbx_like": family_sbx,
        "mean_family_share_de_like": family_de,
        "mean_regime_share_repair": 0.0,
        "mean_regime_share_expand": 0.8,
        "mean_regime_share_refine": 0.2,
        "mean_intent_share_exploratory": intent_exploratory,
        "mean_intent_share_balanced": intent_balanced,
        "mean_intent_share_local_refine": intent_local,
        "mean_intent_share_mutation_heavy": intent_mutation,
        "mean_intent_share_feasibility_biased": intent_feasible,
    }


def _trace_row(
    *,
    suite: str,
    problem: str,
    variant: str,
    run_id: str,
    step_index: int,
    budget_progress: float,
    operator_family: str,
    intent_prototype: str,
) -> dict[str, object]:
    return {
        "suite": suite,
        "run_id": run_id,
        "host": "nsgaii",
        "problem": problem,
        "variant": variant,
        "variant_group": "adaptive",
        "seed": 0,
        "step_index": step_index,
        "generation": step_index,
        "budget_progress": budget_progress,
        "regime": "expand" if budget_progress < 0.66 else "refine",
        "operator_family": operator_family,
        "intent_prototype": intent_prototype,
        "bounded_reward": 0.4 + 0.1 * step_index,
        "overhead_ms": 0.5,
    }


def test_source_attribution_and_benchmark_sensitivity_are_interpretable() -> None:
    summary_rows = [
        _summary_row(suite="zcat", problem="zcat1", variant="fixed_sbx", mean_hv=0.60, mean_time_ms=100.0, family_sbx=1.0, family_de=0.0, intent_exploratory=0.0, intent_balanced=0.0, intent_local=0.0, intent_mutation=0.0),
        _summary_row(suite="zcat", problem="zcat1", variant="fixed_de", mean_hv=0.50, mean_time_ms=110.0, family_sbx=0.0, family_de=1.0, intent_exploratory=0.0, intent_balanced=0.0, intent_local=0.0, intent_mutation=0.0),
        _summary_row(suite="zcat", problem="zcat1", variant="adaptive_flat_operator", mean_hv=0.63, mean_time_ms=118.0, family_sbx=0.4, family_de=0.6, intent_exploratory=0.0, intent_balanced=1.0, intent_local=0.0, intent_mutation=0.0, family_switches=4.0, reward=0.32),
        _summary_row(suite="zcat", problem="zcat1", variant="adaptive_flat_parameter", mean_hv=0.69, mean_time_ms=119.0, family_sbx=1.0, family_de=0.0, intent_exploratory=0.3, intent_balanced=0.2, intent_local=0.3, intent_mutation=0.2, intent_switches=6.0, reward=0.36),
        _summary_row(suite="zcat", problem="zcat1", variant="adaptive_hierarchical_joint", mean_hv=0.72, mean_time_ms=125.0, family_sbx=0.6, family_de=0.4, intent_exploratory=0.3, intent_balanced=0.2, intent_local=0.4, intent_mutation=0.1, family_switches=5.0, intent_switches=7.0, reward=0.40),
        _summary_row(suite="zcat", problem="zcat1", variant="adaptive_hierarchical_joint_no_regime", mean_hv=0.715, mean_time_ms=124.0, family_sbx=0.6, family_de=0.4, intent_exploratory=0.35, intent_balanced=0.2, intent_local=0.35, intent_mutation=0.1, family_switches=5.0, intent_switches=7.0, reward=0.39),
        _summary_row(suite="zcat", problem="zcat1", variant="adaptive_hierarchical_joint_fixed_family_sbx", mean_hv=0.71, mean_time_ms=123.0, family_sbx=1.0, family_de=0.0, intent_exploratory=0.25, intent_balanced=0.2, intent_local=0.4, intent_mutation=0.15, intent_switches=7.0, reward=0.39),
        _summary_row(suite="zcat", problem="zcat1", variant="adaptive_hierarchical_joint_fixed_family_de", mean_hv=0.58, mean_time_ms=126.0, family_sbx=0.0, family_de=1.0, intent_exploratory=0.4, intent_balanced=0.2, intent_local=0.2, intent_mutation=0.2, intent_switches=6.0, reward=0.28),
        _summary_row(suite="anchor", problem="zdt1", variant="fixed_sbx", mean_hv=0.80, mean_time_ms=90.0, family_sbx=1.0, family_de=0.0, intent_exploratory=0.0, intent_balanced=0.0, intent_local=0.0, intent_mutation=0.0),
        _summary_row(suite="anchor", problem="zdt1", variant="fixed_de", mean_hv=0.62, mean_time_ms=92.0, family_sbx=0.0, family_de=1.0, intent_exploratory=0.0, intent_balanced=0.0, intent_local=0.0, intent_mutation=0.0),
        _summary_row(suite="anchor", problem="zdt1", variant="adaptive_flat_operator", mean_hv=0.79, mean_time_ms=98.0, family_sbx=0.5, family_de=0.5, intent_exploratory=0.0, intent_balanced=1.0, intent_local=0.0, intent_mutation=0.0, family_switches=2.0, reward=0.20),
        _summary_row(suite="anchor", problem="zdt1", variant="adaptive_flat_parameter", mean_hv=0.81, mean_time_ms=99.0, family_sbx=1.0, family_de=0.0, intent_exploratory=0.2, intent_balanced=0.3, intent_local=0.4, intent_mutation=0.1, intent_switches=3.0, reward=0.23),
        _summary_row(suite="anchor", problem="zdt1", variant="adaptive_hierarchical_joint", mean_hv=0.815, mean_time_ms=101.0, family_sbx=0.8, family_de=0.2, intent_exploratory=0.2, intent_balanced=0.3, intent_local=0.4, intent_mutation=0.1, family_switches=2.0, intent_switches=3.0, reward=0.24),
        _summary_row(suite="anchor", problem="zdt1", variant="adaptive_hierarchical_joint_no_regime", mean_hv=0.814, mean_time_ms=100.0, family_sbx=0.8, family_de=0.2, intent_exploratory=0.25, intent_balanced=0.3, intent_local=0.35, intent_mutation=0.1, family_switches=2.0, intent_switches=3.0, reward=0.24),
        _summary_row(suite="anchor", problem="zdt1", variant="adaptive_hierarchical_joint_fixed_family_sbx", mean_hv=0.812, mean_time_ms=100.0, family_sbx=1.0, family_de=0.0, intent_exploratory=0.2, intent_balanced=0.25, intent_local=0.45, intent_mutation=0.1, intent_switches=3.0, reward=0.24),
        _summary_row(suite="anchor", problem="zdt1", variant="adaptive_hierarchical_joint_fixed_family_de", mean_hv=0.66, mean_time_ms=102.0, family_sbx=0.0, family_de=1.0, intent_exploratory=0.3, intent_balanced=0.2, intent_local=0.2, intent_mutation=0.3, intent_switches=3.0, reward=0.18),
    ]
    problem_host_rows = compute_suite_problem_host_summary(summary_rows)
    concentration_rows = build_suite_concentration_summary(problem_host_rows)
    phase_rows = build_suite_phase_summary(
        [
            _trace_row(suite="zcat", problem="zcat1", variant="adaptive_hierarchical_joint", run_id="z1", step_index=0, budget_progress=0.1, operator_family="de_like", intent_prototype="exploratory"),
            _trace_row(suite="zcat", problem="zcat1", variant="adaptive_hierarchical_joint", run_id="z1", step_index=1, budget_progress=0.9, operator_family="sbx_like", intent_prototype="local_refine"),
            _trace_row(suite="anchor", problem="zdt1", variant="adaptive_hierarchical_joint", run_id="a1", step_index=0, budget_progress=0.1, operator_family="sbx_like", intent_prototype="balanced"),
            _trace_row(suite="anchor", problem="zdt1", variant="adaptive_hierarchical_joint", run_id="a1", step_index=1, budget_progress=0.9, operator_family="sbx_like", intent_prototype="local_refine"),
        ]
    )
    heterogeneity_rows = build_suite_heterogeneity_summary(problem_host_rows, phase_rows)
    source_rows = build_source_attribution_summary(problem_host_rows)
    benchmark_rows = build_benchmark_sensitivity_summary(source_rows, concentration_rows, heterogeneity_rows)

    proto_row = next(row for row in source_rows if row["suite"] == "overall" and row["comparison_name"] == "adaptive_hierarchical_joint_fixed_family_sbx_vs_best_fixed")
    regime_row = next(row for row in source_rows if row["suite"] == "overall" and row["comparison_name"] == "hierarchical_vs_no_regime")
    family_row = next(row for row in source_rows if row["suite"] == "overall" and row["comparison_name"] == "hierarchical_vs_fixed_family_sbx")
    sensitivity = next(row for row in benchmark_rows if row["metric"] == "hierarchical_mean_hv_gap_vs_best_fixed")

    assert proto_row["mean_hv_gap"] > 0.0
    assert abs(float(regime_row["mean_hv_gap"])) < float(family_row["mean_hv_gap"])
    assert sensitivity["zcat_value"] > sensitivity["anchor_value"]


def test_overhead_profile_summary_and_verdict_capture_control_share() -> None:
    run_rows = [
        {
            "suite": "zcat",
            "host": "nsgaii",
            "variant": "adaptive_hierarchical_joint",
            "time_ms": 100.0,
            "profile_start_step_time_ms": 1.0,
            "profile_router_time_ms": 2.0,
            "profile_policy_select_time_ms": 3.0,
            "profile_policy_update_time_ms": 4.0,
            "profile_decode_time_ms": 5.0,
            "profile_trace_time_ms": 6.0,
            "profile_variation_time_ms": 10.0,
            "profile_evaluation_time_ms": 60.0,
            "profile_survival_time_ms": 5.0,
            "profile_total_runtime_ms": 100.0,
        },
        {
            "suite": "anchor",
            "host": "nsgaii",
            "variant": "adaptive_hierarchical_joint",
            "time_ms": 80.0,
            "profile_start_step_time_ms": 1.0,
            "profile_router_time_ms": 1.0,
            "profile_policy_select_time_ms": 2.0,
            "profile_policy_update_time_ms": 2.0,
            "profile_decode_time_ms": 3.0,
            "profile_trace_time_ms": 2.0,
            "profile_variation_time_ms": 8.0,
            "profile_evaluation_time_ms": 50.0,
            "profile_survival_time_ms": 4.0,
            "profile_total_runtime_ms": 80.0,
        },
    ]
    overhead_rows = build_overhead_profile_summary(run_rows)
    control_row = next(
        row
        for row in overhead_rows
        if row["suite"] == "overall" and row["host"] == "all_hosts" and row["variant"] == "adaptive_hierarchical_joint" and row["component"] == "control_total"
    )
    evaluation_row = next(
        row
        for row in overhead_rows
        if row["suite"] == "overall" and row["host"] == "all_hosts" and row["variant"] == "adaptive_hierarchical_joint" and row["component"] == "evaluation_time"
    )

    verdict = build_ablation_final_verdict(
        configs={"zcat": {"hosts": ["nsgaii"]}, "anchor": {"hosts": ["nsgaii"]}},
        source_attribution_rows=[
            {"suite": "zcat", "comparison_name": "adaptive_hierarchical_joint_vs_best_fixed", "wins": 1, "losses": 0, "cases": 1, "mean_hv_gap": 0.1, "median_runtime_ratio": 1.3},
            {"suite": "anchor", "comparison_name": "adaptive_hierarchical_joint_vs_best_fixed", "wins": 1, "losses": 0, "cases": 1, "mean_hv_gap": 0.02, "median_runtime_ratio": 1.1},
            {"suite": "overall", "comparison_name": "adaptive_hierarchical_joint_fixed_family_sbx_vs_best_fixed", "wins": 2, "losses": 0, "cases": 2, "mean_hv_gap": 0.08},
            {"suite": "overall", "comparison_name": "hierarchical_vs_fixed_family_sbx", "wins": 1, "losses": 1, "cases": 2, "mean_hv_gap": 0.01},
            {"suite": "overall", "comparison_name": "hierarchical_vs_no_regime", "wins": 1, "losses": 1, "cases": 2, "mean_hv_gap": 0.0},
        ],
        benchmark_sensitivity_rows=[{"metric": "hierarchical_mean_hv_gap_vs_best_fixed", "zcat_value": 0.1, "anchor_value": 0.02}],
        overhead_rows=overhead_rows,
        concentration_rows=[
            {"variant": "adaptive_hierarchical_joint", "dominant_family_share": 0.8, "dominant_intent_share": 0.5},
            {"variant": "adaptive_hierarchical_joint", "dominant_family_share": 0.7, "dominant_intent_share": 0.4},
        ],
        heterogeneity_rows=[
            {"suite": "zcat", "metric": "phase_intent_shift_tvd", "variant": "adaptive_hierarchical_joint", "value": 0.3},
            {"suite": "anchor", "metric": "phase_intent_shift_tvd", "variant": "adaptive_hierarchical_joint", "value": 0.1},
        ],
    )

    assert control_row["median_share_of_total_runtime"] > 0.0
    assert evaluation_row["median_share_of_total_runtime"] > control_row["median_share_of_total_runtime"]
    assert verdict["verdict"] == "WEAK_GO_PIVOT_TO_PROTOTYPE_STORY"
