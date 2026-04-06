"""Regenerate summaries and the Markdown report for the GCES ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

from gces_ablation_zcat_2obj_common import (
    DEFAULT_OUTPUT_DIR,
    load_raw_records,
    load_run_config,
    write_analysis_artifacts,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate summary CSVs and Markdown report for the GCES ablation campaign.")
    parser.add_argument("--input-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory containing raw_results.csv and run_config.json.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    records = load_raw_records(input_dir / "raw_results.csv")
    run_config = load_run_config(input_dir / "run_config.json")
    write_analysis_artifacts(
        output_dir=input_dir,
        records=records,
        run_config=run_config,
        alpha=float(run_config["alpha"]),
        tie_tol=float(run_config["tie_tol"]),
    )
    print(f"Regenerated summary.csv, comparison.csv, and report.md in {input_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
