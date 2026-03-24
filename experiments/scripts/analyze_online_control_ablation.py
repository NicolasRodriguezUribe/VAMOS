from __future__ import annotations

import argparse
from pathlib import Path

from vamos.experiment.online_control_ablation_analysis import (
    build_ablation_analysis_report,
    build_ablation_final_verdict,
    build_ablation_policy_comparison,
    build_benchmark_sensitivity_summary,
    build_overhead_profile_summary,
    build_source_attribution_summary,
    build_suite_concentration_summary,
    build_suite_heterogeneity_summary,
    build_suite_phase_summary,
    compute_suite_problem_host_summary,
    load_ablation_outputs,
    render_ablation_findings_memo,
    write_ablation_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ZCAT-first online-control ablation outputs.")
    parser.add_argument("--zcat-dir", type=Path, required=True, help="Output directory for the ZCAT-first ablation pilot.")
    parser.add_argument("--anchor-dir", type=Path, required=True, help="Output directory for the anchor-suite ablation pilot.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "online_control_ablation_analysis",
        help="Output directory for the ablation analysis artifacts.",
    )
    args = parser.parse_args()

    run_rows, summary_rows, trace_rows, configs = load_ablation_outputs(zcat_dir=args.zcat_dir, anchor_dir=args.anchor_dir)
    problem_host_rows = compute_suite_problem_host_summary(summary_rows)
    comparison_rows = build_ablation_policy_comparison(problem_host_rows)
    source_attribution_rows = build_source_attribution_summary(problem_host_rows)
    concentration_rows = build_suite_concentration_summary(problem_host_rows)
    phase_rows = build_suite_phase_summary(trace_rows)
    heterogeneity_rows = build_suite_heterogeneity_summary(problem_host_rows, phase_rows)
    overhead_rows = build_overhead_profile_summary(run_rows)
    benchmark_sensitivity_rows = build_benchmark_sensitivity_summary(
        source_attribution_rows,
        concentration_rows,
        heterogeneity_rows,
    )
    analysis_report = build_ablation_analysis_report(
        configs=configs,
        source_attribution_rows=source_attribution_rows,
        benchmark_sensitivity_rows=benchmark_sensitivity_rows,
        overhead_rows=overhead_rows,
        concentration_rows=concentration_rows,
        heterogeneity_rows=heterogeneity_rows,
    )
    final_verdict = build_ablation_final_verdict(
        configs=configs,
        source_attribution_rows=source_attribution_rows,
        benchmark_sensitivity_rows=benchmark_sensitivity_rows,
        overhead_rows=overhead_rows,
        concentration_rows=concentration_rows,
        heterogeneity_rows=heterogeneity_rows,
    )
    memo_text = render_ablation_findings_memo(
        configs=configs,
        source_attribution_rows=source_attribution_rows,
        benchmark_sensitivity_rows=benchmark_sensitivity_rows,
        overhead_rows=overhead_rows,
        concentration_rows=concentration_rows,
        heterogeneity_rows=heterogeneity_rows,
        verdict_payload=final_verdict,
    )

    write_ablation_outputs(
        output_dir=args.output,
        comparison_rows=comparison_rows,
        source_attribution_rows=source_attribution_rows,
        overhead_rows=overhead_rows,
        benchmark_sensitivity_rows=benchmark_sensitivity_rows,
        analysis_report=analysis_report,
        memo_text=memo_text,
        final_verdict=final_verdict,
    )
    print(f"[ablation-analysis] wrote {args.output / 'ablation_policy_comparison.csv'}")
    print(f"[ablation-analysis] wrote {args.output / 'source_attribution_summary.csv'}")
    print(f"[ablation-analysis] wrote {args.output / 'overhead_profile_summary.csv'}")
    print(f"[ablation-analysis] wrote {args.output / 'benchmark_sensitivity_summary.csv'}")
    print(f"[ablation-analysis] wrote {args.output / 'ablation_analysis_report.json'}")
    print(f"[ablation-analysis] wrote {args.output / 'ablation_findings_memo.md'}")
    print(f"[ablation-analysis] wrote {args.output / 'ablation_final_verdict.json'}")
    print(f"[ablation-analysis] verdict={final_verdict['verdict']}")


if __name__ == "__main__":
    main()
