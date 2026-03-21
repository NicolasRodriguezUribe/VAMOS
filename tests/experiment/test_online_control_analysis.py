from __future__ import annotations

from vamos.experiment.online_control_analysis import (
    build_concentration_summary,
    build_go_no_go_analysis,
    build_phase_summary,
    build_policy_comparison,
    compute_problem_host_summary,
    phase_from_progress,
)


def _summary_rows() -> list[dict[str, object]]:
    return [
        {
            "host": "nsgaii",
            "problem": "zdt1",
            "variant": "fixed_sbx",
            "variant_group": "fixed",
            "mean_hv": 1.0,
            "mean_igd_plus": 1.2,
            "mean_time_ms": 10.0,
            "mean_average_reward": 0.0,
            "mean_family_concentration": 1.0,
            "mean_regime_concentration": 0.0,
            "mean_intent_concentration": 0.0,
            "mean_family_switches": 0.0,
            "mean_regime_switches": 0.0,
            "mean_intent_switches": 0.0,
            "mean_family_share_sbx_like": 1.0,
            "mean_family_share_de_like": 0.0,
            "mean_regime_share_repair": 0.0,
            "mean_regime_share_expand": 0.0,
            "mean_regime_share_refine": 0.0,
            "mean_intent_share_exploratory": 0.0,
            "mean_intent_share_balanced": 0.0,
            "mean_intent_share_local_refine": 0.0,
            "mean_intent_share_mutation_heavy": 0.0,
            "mean_intent_share_feasibility_biased": 0.0,
        },
        {
            "host": "nsgaii",
            "problem": "zdt1",
            "variant": "fixed_de",
            "variant_group": "fixed",
            "mean_hv": 1.3,
            "mean_igd_plus": 1.1,
            "mean_time_ms": 11.0,
            "mean_average_reward": 0.0,
            "mean_family_concentration": 1.0,
            "mean_regime_concentration": 0.0,
            "mean_intent_concentration": 0.0,
            "mean_family_switches": 0.0,
            "mean_regime_switches": 0.0,
            "mean_intent_switches": 0.0,
            "mean_family_share_sbx_like": 0.0,
            "mean_family_share_de_like": 1.0,
            "mean_regime_share_repair": 0.0,
            "mean_regime_share_expand": 0.0,
            "mean_regime_share_refine": 0.0,
            "mean_intent_share_exploratory": 0.0,
            "mean_intent_share_balanced": 0.0,
            "mean_intent_share_local_refine": 0.0,
            "mean_intent_share_mutation_heavy": 0.0,
            "mean_intent_share_feasibility_biased": 0.0,
        },
        {
            "host": "nsgaii",
            "problem": "zdt1",
            "variant": "adaptive_flat_operator",
            "variant_group": "adaptive",
            "mean_hv": 1.4,
            "mean_igd_plus": 1.0,
            "mean_time_ms": 12.0,
            "mean_average_reward": 0.4,
            "mean_family_concentration": 0.7,
            "mean_regime_concentration": 0.5,
            "mean_intent_concentration": 1.0,
            "mean_family_switches": 2.0,
            "mean_regime_switches": 1.0,
            "mean_intent_switches": 0.0,
            "mean_family_share_sbx_like": 0.2,
            "mean_family_share_de_like": 0.8,
            "mean_regime_share_repair": 0.0,
            "mean_regime_share_expand": 0.6,
            "mean_regime_share_refine": 0.4,
            "mean_intent_share_exploratory": 0.0,
            "mean_intent_share_balanced": 1.0,
            "mean_intent_share_local_refine": 0.0,
            "mean_intent_share_mutation_heavy": 0.0,
            "mean_intent_share_feasibility_biased": 0.0,
        },
        {
            "host": "nsgaii",
            "problem": "zdt1",
            "variant": "adaptive_flat_parameter",
            "variant_group": "adaptive",
            "mean_hv": 1.35,
            "mean_igd_plus": 1.05,
            "mean_time_ms": 11.5,
            "mean_average_reward": 0.35,
            "mean_family_concentration": 1.0,
            "mean_regime_concentration": 0.55,
            "mean_intent_concentration": 0.6,
            "mean_family_switches": 0.0,
            "mean_regime_switches": 1.0,
            "mean_intent_switches": 1.5,
            "mean_family_share_sbx_like": 1.0,
            "mean_family_share_de_like": 0.0,
            "mean_regime_share_repair": 0.0,
            "mean_regime_share_expand": 0.5,
            "mean_regime_share_refine": 0.5,
            "mean_intent_share_exploratory": 0.3,
            "mean_intent_share_balanced": 0.1,
            "mean_intent_share_local_refine": 0.4,
            "mean_intent_share_mutation_heavy": 0.2,
            "mean_intent_share_feasibility_biased": 0.0,
        },
        {
            "host": "nsgaii",
            "problem": "zdt1",
            "variant": "adaptive_hierarchical_joint",
            "variant_group": "adaptive",
            "mean_hv": 1.5,
            "mean_igd_plus": 0.95,
            "mean_time_ms": 12.5,
            "mean_average_reward": 0.45,
            "mean_family_concentration": 0.8,
            "mean_regime_concentration": 0.6,
            "mean_intent_concentration": 0.7,
            "mean_family_switches": 2.5,
            "mean_regime_switches": 1.2,
            "mean_intent_switches": 2.0,
            "mean_family_share_sbx_like": 0.3,
            "mean_family_share_de_like": 0.7,
            "mean_regime_share_repair": 0.0,
            "mean_regime_share_expand": 0.55,
            "mean_regime_share_refine": 0.45,
            "mean_intent_share_exploratory": 0.5,
            "mean_intent_share_balanced": 0.2,
            "mean_intent_share_local_refine": 0.2,
            "mean_intent_share_mutation_heavy": 0.1,
            "mean_intent_share_feasibility_biased": 0.0,
        },
    ]


def _trace_rows() -> list[dict[str, object]]:
    return [
        {
            "run_id": "r1",
            "host": "nsgaii",
            "problem": "zdt1",
            "variant": "adaptive_hierarchical_joint",
            "step_index": 0,
            "budget_progress": 0.1,
            "regime": "expand",
            "operator_family": "de_like",
            "intent_prototype": "exploratory",
            "bounded_reward": 0.2,
            "overhead_ms": 0.4,
        },
        {
            "run_id": "r1",
            "host": "nsgaii",
            "problem": "zdt1",
            "variant": "adaptive_hierarchical_joint",
            "step_index": 1,
            "budget_progress": 0.2,
            "regime": "expand",
            "operator_family": "de_like",
            "intent_prototype": "exploratory",
            "bounded_reward": 0.3,
            "overhead_ms": 0.4,
        },
        {
            "run_id": "r1",
            "host": "nsgaii",
            "problem": "zdt1",
            "variant": "adaptive_hierarchical_joint",
            "step_index": 2,
            "budget_progress": 0.5,
            "regime": "refine",
            "operator_family": "sbx_like",
            "intent_prototype": "local_refine",
            "bounded_reward": 0.4,
            "overhead_ms": 0.4,
        },
        {
            "run_id": "r1",
            "host": "nsgaii",
            "problem": "zdt1",
            "variant": "adaptive_hierarchical_joint",
            "step_index": 3,
            "budget_progress": 0.9,
            "regime": "refine",
            "operator_family": "sbx_like",
            "intent_prototype": "local_refine",
            "bounded_reward": 0.5,
            "overhead_ms": 0.4,
        },
    ]


def test_phase_from_progress_splits_early_mid_late() -> None:
    assert phase_from_progress(0.1) == "early"
    assert phase_from_progress(0.5) == "mid"
    assert phase_from_progress(0.9) == "late"


def test_compute_problem_host_summary_uses_best_fixed_baseline() -> None:
    rows = compute_problem_host_summary(_summary_rows())
    hierarchical = next(row for row in rows if row["variant"] == "adaptive_hierarchical_joint")

    assert hierarchical["best_fixed_variant"] == "fixed_de"
    assert hierarchical["comparison_to_best_fixed"] == "win"
    assert hierarchical["hv_gap_vs_best_fixed"] > 0.0
    assert hierarchical["runtime_ratio_vs_best_fixed"] > 1.0


def test_build_policy_comparison_compares_hierarchical_against_flats() -> None:
    rows = compute_problem_host_summary(_summary_rows())
    comparisons = build_policy_comparison(rows)
    target = next(
        row
        for row in comparisons
        if row["left_variant"] == "adaptive_hierarchical_joint" and row["right_variant"] == "adaptive_flat_operator"
    )

    assert target["outcome"] == "win"
    assert target["hv_gap"] > 0.0


def test_build_concentration_summary_emits_dominant_share_metrics() -> None:
    rows = compute_problem_host_summary(_summary_rows())
    concentration = build_concentration_summary(rows)
    hierarchical = next(row for row in concentration if row["variant"] == "adaptive_hierarchical_joint")

    assert hierarchical["dominant_family_share"] == 0.7
    assert hierarchical["family_entropy"] > 0.0


def test_build_phase_summary_groups_trace_rows_by_phase() -> None:
    phase_rows = build_phase_summary(_trace_rows())
    early = next(row for row in phase_rows if row["phase"] == "early")
    late = next(row for row in phase_rows if row["phase"] == "late")

    assert early["family_share_de_like"] == 1.0
    assert late["family_share_sbx_like"] == 1.0
    assert late["mean_reward"] > early["mean_reward"]


def test_build_go_no_go_analysis_reports_core_checks() -> None:
    problem_host_rows = compute_problem_host_summary(_summary_rows())
    comparisons = build_policy_comparison(problem_host_rows)
    concentration = build_concentration_summary(problem_host_rows)
    go_no_go = build_go_no_go_analysis(problem_host_rows, comparisons, concentration, [], transfer_summary=[])

    assert "adaptive_hierarchical_beats_best_fixed" in go_no_go["checks"]
    assert "overall_decision" in go_no_go
