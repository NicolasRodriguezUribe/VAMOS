"""Regenerate summaries and report for the 3-objective ZCAT difficulty campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

from farthest_gces_difficulty_ab_zcat_3obj_common import load_raw_records, write_analysis_artifacts
from gces_ablation_zcat_2obj_common import load_run_config

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "experiments" / "farthest_gces_difficulty_ab_zcat_3obj_full"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate summaries and report for the 3-objective ZCAT difficulty robustness campaign."
    )
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
    print(f"Regenerated summary/report artifacts in {input_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
