from __future__ import annotations

import argparse
from pathlib import Path

from canonical_runs import run_rows, write_tidy_csv

CORE_COLUMNS = [
    "run_path",
    "run_id",
    "task_id",
    "status",
    "campaign",
    "variant",
    "suite",
    "algorithm",
    "engine",
    "problem",
    "seed",
    "max_evaluations",
    "population_size",
    "n_obj",
    "n_var",
    "runtime_seconds",
    "front_size",
    "objective_count",
    "decision_rows",
    "decision_columns",
    "git_revision",
    "timestamp",
    "vamos_version",
    "config_keys",
]


def collect_campaign(results_root: Path, *, campaign: str) -> list[dict[str, object]]:
    return run_rows(results_root, campaign=campaign)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, help="Campaign label written to the derived table")
    parser.add_argument("--results-root", required=True, help="Directory containing canonical run manifests")
    parser.add_argument("--out", default=None, help="Derived tidy CSV path")
    parser.add_argument("--sample-out", default=None, help="Derived sample CSV path")
    parser.add_argument("--sample-n", type=int, default=12)
    args = parser.parse_args()

    repo = Path.cwd()
    results_root = (repo / args.results_root).resolve()
    if not results_root.exists():
        print("ERROR: results root not found:", results_root)
        return 2

    rows = collect_campaign(results_root, campaign=args.campaign)
    if not rows:
        print("ERROR: no canonical runs found.")
        return 3

    output = (repo / (args.out or f"artifacts/tidy/{args.campaign}.csv")).resolve()
    sample = (repo / (args.sample_out or f"experiments/sample_outputs/{args.campaign}_sample.csv")).resolve()
    columns = write_tidy_csv(output, rows, core_columns=CORE_COLUMNS)
    write_tidy_csv(sample, rows[: args.sample_n], core_columns=CORE_COLUMNS)

    print("Wrote:", output, "rows:", len(rows), "columns:", len(columns))
    print("Wrote derived sample:", sample, "rows:", min(len(rows), args.sample_n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
