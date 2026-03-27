from __future__ import annotations

from vamos.experiment.semantic_prototype_confirmatory_analysis import (
    build_confirmatory_case_comparisons,
    build_confirmatory_final_verdict,
    build_confirmatory_report,
    build_confirmatory_summary,
    build_final_confirmatory_report,
    build_final_confirmatory_tables,
    build_overhead_view,
    build_phase_diagnostics,
    build_prototype_profile,
    render_final_confirmatory_findings_memo,
)


def _summary_row(
    *,
    suite: str,
    host: str,
    problem: str,
    variant: str,
    mean_hv: float,
    mean_time_ms: float,
    reward: float,
    intent_exploratory: float,
    intent_balanced: float,
    intent_local_refine: float,
    intent_mutation_heavy: float,
) -> dict[str, object]:
    return {
        "suite": suite,
        "host": host,
        "problem": problem,
        "variant": variant,
        "variant_group": "fixed" if variant.startswith("fixed_") else "adaptive",
        "mean_hv": mean_hv,
        "mean_igd_plus": 1.0 / max(mean_hv, 0.1),
        "mean_time_ms": mean_time_ms,
        "mean_average_reward": reward,
        "mean_average_overhead_ms": 1.0,
        "mean_family_concentration": 1.0,
        "mean_regime_concentration": 0.95,
        "mean_intent_concentration": sum(
            share * share for share in (intent_exploratory, intent_balanced, intent_local_refine, intent_mutation_heavy)
        ),
        "mean_family_switches": 0.0,
        "mean_regime_switches": 1.0,
        "mean_intent_switches": 4.0 if variant == "semantic_prototype_sbx" else 2.0,
        "mean_family_share_sbx_like": 1.0,
        "mean_family_share_de_like": 0.0,
        "mean_regime_share_repair": 0.0,
        "mean_regime_share_expand": 0.8,
        "mean_regime_share_refine": 0.2,
        "mean_intent_share_exploratory": intent_exploratory,
        "mean_intent_share_balanced": intent_balanced,
        "mean_intent_share_local_refine": intent_local_refine,
        "mean_intent_share_mutation_heavy": intent_mutation_heavy,
        "mean_intent_share_feasibility_biased": 0.0,
    }


def _trace_row(
    *,
    suite: str,
    host: str,
    problem: str,
    run_id: str,
    step_index: int,
    budget_progress: float,
    intent_prototype: str,
) -> dict[str, object]:
    return {
        "suite": suite,
        "run_id": run_id,
        "host": host,
        "problem": problem,
        "variant": "semantic_prototype_sbx",
        "variant_group": "adaptive",
        "seed": 0,
        "step_index": step_index,
        "generation": step_index,
        "budget_progress": budget_progress,
        "regime": "expand" if budget_progress < 0.66 else "refine",
        "operator_family": "sbx_like",
        "intent_prototype": intent_prototype,
        "bounded_reward": 0.3 + 0.1 * step_index,
        "overhead_ms": 0.5,
    }


def _run_row(*, suite: str, host: str, variant: str, time_ms: float) -> dict[str, object]:
    return {
        "suite": suite,
        "host": host,
        "variant": variant,
        "time_ms": time_ms,
        "profile_start_step_time_ms": 1.0 if variant != "fixed_sbx" else 0.0,
        "profile_router_time_ms": 1.0 if variant != "fixed_sbx" else 0.0,
        "profile_policy_select_time_ms": 2.0 if variant != "fixed_sbx" else 0.0,
        "profile_policy_update_time_ms": 2.0 if variant != "fixed_sbx" else 0.0,
        "profile_decode_time_ms": 3.0 if variant != "fixed_sbx" else 0.0,
        "profile_trace_time_ms": 1.0 if variant != "fixed_sbx" else 0.0,
        "profile_variation_time_ms": 10.0,
        "profile_evaluation_time_ms": 55.0,
        "profile_survival_time_ms": 12.0,
        "profile_total_runtime_ms": time_ms,
    }


def test_confirmatory_analysis_tracks_prototype_story() -> None:
    summary_rows = [
        _summary_row(suite="zcat", host="nsgaii", problem="zcat1", variant="fixed_sbx", mean_hv=0.60, mean_time_ms=100.0, reward=0.0, intent_exploratory=0.0, intent_balanced=0.0, intent_local_refine=0.0, intent_mutation_heavy=0.0),
        _summary_row(suite="zcat", host="nsgaii", problem="zcat1", variant="semantic_prototype_sbx", mean_hv=0.74, mean_time_ms=135.0, reward=0.42, intent_exploratory=0.35, intent_balanced=0.20, intent_local_refine=0.35, intent_mutation_heavy=0.10),
        _summary_row(suite="zcat", host="nsgaii", problem="zcat1", variant="adaptive_hierarchical_joint", mean_hv=0.70, mean_time_ms=138.0, reward=0.39, intent_exploratory=0.30, intent_balanced=0.25, intent_local_refine=0.35, intent_mutation_heavy=0.10),
        _summary_row(suite="zcat", host="nsgaii", problem="zcat1", variant="adaptive_hierarchical_joint_no_regime", mean_hv=0.69, mean_time_ms=136.0, reward=0.38, intent_exploratory=0.32, intent_balanced=0.24, intent_local_refine=0.34, intent_mutation_heavy=0.10),
        _summary_row(suite="anchor", host="moead", problem="zdt1", variant="fixed_sbx", mean_hv=0.80, mean_time_ms=90.0, reward=0.0, intent_exploratory=0.0, intent_balanced=0.0, intent_local_refine=0.0, intent_mutation_heavy=0.0),
        _summary_row(suite="anchor", host="moead", problem="zdt1", variant="semantic_prototype_sbx", mean_hv=0.83, mean_time_ms=120.0, reward=0.26, intent_exploratory=0.22, intent_balanced=0.28, intent_local_refine=0.38, intent_mutation_heavy=0.12),
        _summary_row(suite="anchor", host="moead", problem="zdt1", variant="adaptive_hierarchical_joint", mean_hv=0.82, mean_time_ms=122.0, reward=0.25, intent_exploratory=0.22, intent_balanced=0.30, intent_local_refine=0.36, intent_mutation_heavy=0.12),
        _summary_row(suite="anchor", host="moead", problem="zdt1", variant="adaptive_hierarchical_joint_no_regime", mean_hv=0.819, mean_time_ms=121.0, reward=0.25, intent_exploratory=0.24, intent_balanced=0.30, intent_local_refine=0.34, intent_mutation_heavy=0.12),
    ]
    trace_rows = [
        _trace_row(suite="zcat", host="nsgaii", problem="zcat1", run_id="z1", step_index=0, budget_progress=0.1, intent_prototype="exploratory"),
        _trace_row(suite="zcat", host="nsgaii", problem="zcat1", run_id="z1", step_index=1, budget_progress=0.9, intent_prototype="local_refine"),
        _trace_row(suite="anchor", host="moead", problem="zdt1", run_id="a1", step_index=0, budget_progress=0.1, intent_prototype="balanced"),
        _trace_row(suite="anchor", host="moead", problem="zdt1", run_id="a1", step_index=1, budget_progress=0.9, intent_prototype="local_refine"),
    ]
    run_rows = [
        _run_row(suite="zcat", host="nsgaii", variant="semantic_prototype_sbx", time_ms=135.0),
        _run_row(suite="zcat", host="nsgaii", variant="fixed_sbx", time_ms=100.0),
        _run_row(suite="zcat", host="nsgaii", variant="adaptive_hierarchical_joint", time_ms=138.0),
        _run_row(suite="anchor", host="moead", variant="semantic_prototype_sbx", time_ms=120.0),
        _run_row(suite="anchor", host="moead", variant="fixed_sbx", time_ms=90.0),
        _run_row(suite="anchor", host="moead", variant="adaptive_hierarchical_joint", time_ms=122.0),
    ]

    case_rows = build_confirmatory_case_comparisons(summary_rows)
    confirmatory_summary = build_confirmatory_summary(case_rows)
    prototype_profile = build_prototype_profile(summary_rows)
    phase_diagnostics = build_phase_diagnostics(trace_rows)
    overhead_view = build_overhead_view(run_rows)
    final_tables = build_final_confirmatory_tables(confirmatory_summary)
    report = build_confirmatory_report(
        config={"hosts": ["nsgaii", "moead"]},
        confirmatory_summary_rows=confirmatory_summary,
        prototype_profile=prototype_profile,
        phase_diagnostics=phase_diagnostics,
        overhead_view=overhead_view,
    )
    verdict = build_confirmatory_final_verdict(
        config={"hosts": ["nsgaii", "moead"]},
        confirmatory_summary_rows=confirmatory_summary,
        prototype_profile=prototype_profile,
        phase_diagnostics=phase_diagnostics,
        overhead_view=overhead_view,
    )
    final_report = build_final_confirmatory_report(
        config={"hosts": ["nsgaii", "moead"]},
        confirmatory_summary_rows=confirmatory_summary,
        compact_tables=final_tables,
        prototype_profile=prototype_profile,
        phase_diagnostics=phase_diagnostics,
        overhead_view=overhead_view,
        final_verdict=verdict,
    )
    memo_text = render_final_confirmatory_findings_memo(
        config={"hosts": ["nsgaii", "moead"], "problems": [{"key": "zcat1", "suite": "zcat"}, {"key": "zdt1", "suite": "anchor"}], "variants": ["fixed_sbx", "semantic_prototype_sbx", "adaptive_hierarchical_joint", "adaptive_hierarchical_joint_no_regime"], "seeds": [0], "population_size": 8, "max_evaluations": 24},
        confirmatory_summary_rows=confirmatory_summary,
        compact_tables=final_tables,
        prototype_profile=prototype_profile,
        phase_diagnostics=phase_diagnostics,
        overhead_view=overhead_view,
        final_verdict=verdict,
    )

    overall_vs_fixed = next(
        row
        for row in confirmatory_summary
        if row["scope_type"] == "overall" and row["comparison_name"] == "semantic_prototype_sbx_vs_fixed_sbx"
    )
    overall_vs_hier = next(
        row
        for row in confirmatory_summary
        if row["scope_type"] == "overall" and row["comparison_name"] == "semantic_prototype_sbx_vs_adaptive_hierarchical_joint"
    )

    assert overall_vs_fixed["wins"] == 2
    assert overall_vs_fixed["mean_hv_gap"] > 0.0
    assert overall_vs_hier["mean_hv_gap"] > 0.0
    assert any(row["scope"] == "overall" and row["comparison_name"] == "semantic_prototype_sbx_vs_fixed_sbx" for row in final_tables)
    assert report["prototype_profile"]["overall"]["median_dominant_intent_share"] > 0.0
    assert final_report["compact_tables"]
    assert report["phase_dynamics"]["shift_summary"]["overall"]["mean_intent_shift_tvd"] > 0.0
    assert report["overhead"]["prototype_sbx"]["host_pipeline_total"]["median_share_of_total_runtime"] > report["overhead"]["prototype_sbx"]["control_total"]["median_share_of_total_runtime"]
    assert "## 8. Final recommendation" in memo_text
    assert verdict["verdict"] == "GO_SWERVO_STYLE"
