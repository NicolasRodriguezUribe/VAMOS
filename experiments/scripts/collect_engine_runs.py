from __future__ import annotations

import argparse
from pathlib import Path

from canonical_runs import write_tidy_csv
from collect_campaign_runs import CORE_COLUMNS, collect_campaign


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/bench_smoke_engines", help="Canonical StudyManifest directory")
    parser.add_argument("--output", default="artifacts/tidy/engine_smoke.csv", help="Derived tidy CSV path")
    parser.add_argument("--campaign", default="bench_smoke_engines", help="Campaign label written to the table")
    args = parser.parse_args()

    repo = Path.cwd()
    input_root = (repo / args.input).resolve()
    output = (repo / args.output).resolve()
    if not input_root.exists():
        print("ERROR: input path not found:", input_root)
        return 2

    rows = collect_campaign(input_root, campaign=args.campaign)
    if not rows:
        print("ERROR: canonical study has no tasks:", input_root)
        return 3

    columns = write_tidy_csv(output, rows, core_columns=CORE_COLUMNS)
    print("Wrote:", output)
    print("Runs:", len(rows))
    print("Columns:", len(columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
