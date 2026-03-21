from __future__ import annotations

import argparse
import json
from pathlib import Path

from vamos.experiment.online_control_analysis import (
    build_analysis_report,
    build_concentration_summary,
    build_go_no_go_analysis,
    build_heterogeneity_summary,
    build_phase_summary,
    build_policy_comparison,
    compute_problem_host_summary,
    load_pilot_output,
    read_csv_rows,
    write_csv_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze online-control pilot outputs.")
    parser.add_argument("output_dir", type=Path, help="Pilot output directory containing runs.csv, summary.csv, and trace_rows.csv.")
    parser.add_argument(
        "--transfer-dir",
        type=Path,
        default=None,
        help="Optional transfer output directory containing transfer_summary.csv for the go/no-go analysis.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    _runs, summary_rows, trace_rows = load_pilot_output(output_dir)
    transfer_summary = read_csv_rows(args.transfer_dir / "transfer_summary.csv") if args.transfer_dir is not None else []

    problem_host_rows = compute_problem_host_summary(summary_rows)
    policy_comparison = build_policy_comparison(problem_host_rows)
    concentration_rows = build_concentration_summary(problem_host_rows)
    phase_rows = build_phase_summary(trace_rows)
    heterogeneity_rows = build_heterogeneity_summary(problem_host_rows, phase_rows)
    analysis_report = build_analysis_report(
        problem_host_rows,
        policy_comparison,
        concentration_rows,
        heterogeneity_rows,
        phase_rows,
        transfer_summary=transfer_summary,
    )
    go_no_go = build_go_no_go_analysis(
        problem_host_rows,
        policy_comparison,
        concentration_rows,
        heterogeneity_rows,
        transfer_summary=transfer_summary,
    )

    write_csv_rows(output_dir / "policy_comparison.csv", policy_comparison)
    write_csv_rows(output_dir / "problem_host_summary.csv", problem_host_rows)
    write_csv_rows(output_dir / "concentration_summary.csv", concentration_rows)
    write_csv_rows(output_dir / "heterogeneity_summary.csv", heterogeneity_rows)
    write_csv_rows(output_dir / "phase_summary.csv", phase_rows)
    (output_dir / "analysis_report.json").write_text(json.dumps(analysis_report, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "go_no_go_analysis.json").write_text(json.dumps(go_no_go, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[analysis] wrote {output_dir / 'policy_comparison.csv'}")
    print(f"[analysis] wrote {output_dir / 'problem_host_summary.csv'}")
    print(f"[analysis] wrote {output_dir / 'concentration_summary.csv'}")
    print(f"[analysis] wrote {output_dir / 'heterogeneity_summary.csv'}")
    print(f"[analysis] wrote {output_dir / 'phase_summary.csv'}")
    print(f"[analysis] wrote {output_dir / 'analysis_report.json'}")
    print(f"[analysis] wrote {output_dir / 'go_no_go_analysis.json'}")
    print(f"[analysis] decision={go_no_go['overall_decision']}")


if __name__ == "__main__":
    main()
