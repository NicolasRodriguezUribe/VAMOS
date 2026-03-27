from __future__ import annotations

import argparse
from pathlib import Path

from vamos.experiment.semantic_prototype_confirmatory_analysis import (
    build_confirmatory_case_comparisons,
    build_confirmatory_final_verdict,
    build_confirmatory_summary,
    build_final_confirmatory_report,
    build_final_confirmatory_tables,
    build_overhead_view,
    build_phase_diagnostics,
    build_prototype_profile,
    load_confirmatory_output,
    render_final_confirmatory_findings_memo,
    write_final_confirmatory_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final paper-facing semantic-prototype-SBX confirmatory package.")
    parser.add_argument("input_dir", type=Path, help="Pilot output directory to analyze.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for final confirmatory artifacts. Defaults to the input directory.",
    )
    args = parser.parse_args()

    output_dir = args.output or args.input_dir
    run_rows, summary_rows, trace_rows, config = load_confirmatory_output(args.input_dir)
    case_rows = build_confirmatory_case_comparisons(summary_rows)
    final_summary_rows = build_confirmatory_summary(case_rows)
    final_tables_rows = build_final_confirmatory_tables(final_summary_rows)
    prototype_profile = build_prototype_profile(summary_rows)
    phase_diagnostics = build_phase_diagnostics(trace_rows)
    overhead_view = build_overhead_view(run_rows)
    final_verdict = build_confirmatory_final_verdict(
        config=config,
        confirmatory_summary_rows=final_summary_rows,
        prototype_profile=prototype_profile,
        phase_diagnostics=phase_diagnostics,
        overhead_view=overhead_view,
    )
    final_report = build_final_confirmatory_report(
        config=config,
        confirmatory_summary_rows=final_summary_rows,
        compact_tables=final_tables_rows,
        prototype_profile=prototype_profile,
        phase_diagnostics=phase_diagnostics,
        overhead_view=overhead_view,
        final_verdict=final_verdict,
    )
    memo_text = render_final_confirmatory_findings_memo(
        config=config,
        confirmatory_summary_rows=final_summary_rows,
        compact_tables=final_tables_rows,
        prototype_profile=prototype_profile,
        phase_diagnostics=phase_diagnostics,
        overhead_view=overhead_view,
        final_verdict=final_verdict,
    )
    write_final_confirmatory_outputs(
        output_dir=output_dir,
        final_summary_rows=final_summary_rows,
        final_tables_rows=final_tables_rows,
        final_report=final_report,
        memo_text=memo_text,
        final_verdict=final_verdict,
    )
    print(f"[final-confirmatory-analysis] wrote {output_dir / 'final_confirmatory_summary.csv'}")
    print(f"[final-confirmatory-analysis] wrote {output_dir / 'final_confirmatory_tables.csv'}")
    print(f"[final-confirmatory-analysis] wrote {output_dir / 'final_confirmatory_report.json'}")
    print(f"[final-confirmatory-analysis] wrote {output_dir / 'final_confirmatory_findings_memo.md'}")
    print(f"[final-confirmatory-analysis] wrote {output_dir / 'final_confirmatory_verdict.json'}")
    print(f"[final-confirmatory-analysis] verdict={final_verdict['verdict']}")


if __name__ == "__main__":
    main()
