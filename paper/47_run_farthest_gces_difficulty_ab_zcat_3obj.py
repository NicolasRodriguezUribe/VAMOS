"""Run the 3-objective ZCAT difficulty robustness campaign for farthest + GCES."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from farthest_gces_difficulty_ab_zcat_3obj_common import (
    ALGORITHM_LABELS,
    DEFAULT_ALGORITHMS,
    DEFAULT_COMPARISONS,
    DEFAULT_DIFFICULTY_CELLS,
    DEFAULT_ENGINE,
    DEFAULT_EVALS,
    DEFAULT_N_OBJ,
    DEFAULT_N_VAR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POP_SIZE,
    DEFAULT_PROBLEMS,
    DEFAULT_SEEDS,
    resolve_worker_count,
    run_campaign,
    select_difficulty_cells,
    write_analysis_artifacts,
    write_csv,
    write_run_config,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 3-objective ZCAT difficulty robustness campaign for farthest + GCES."
    )
    parser.add_argument("--config-ids", nargs="*", default=None, help="Optional subset of difficulty config IDs.")
    parser.add_argument("--problems", nargs="+", default=DEFAULT_PROBLEMS, help="Problem keys to run.")
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS, help="Algorithm IDs to run.")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="Number of seeds to run, starting from 0.")
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE, help="Population size.")
    parser.add_argument("--max-evaluations", type=int, default=DEFAULT_EVALS, help="Evaluation budget per run.")
    parser.add_argument("--n-var", type=int, default=DEFAULT_N_VAR, help="Number of decision variables.")
    parser.add_argument("--n-obj", type=int, default=DEFAULT_N_OBJ, help="Number of objectives.")
    parser.add_argument("--engine", default=DEFAULT_ENGINE, help="VAMOS engine/backend.")
    parser.add_argument("--workers", type=int, default=None, help="Parallel worker count. Defaults to min(cpu_count, 18).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where outputs are written.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level used for Holm-corrected Wilcoxon flags.")
    parser.add_argument("--tie-tol", type=float, default=1e-12, help="Absolute tolerance used to count paired ties.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cells = select_difficulty_cells(args.config_ids)
    seeds = list(range(int(args.seeds)))
    workers = resolve_worker_count(args.workers)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    comparisons = [tuple(pair) for pair in DEFAULT_COMPARISONS if pair[0] in args.algorithms and pair[1] in args.algorithms]
    expected_total_runs = len(cells) * len(args.problems) * len(args.algorithms) * len(seeds)
    run_config = {
        "difficulty_cells": [asdict(cell) for cell in cells],
        "problems": list(args.problems),
        "algorithms": list(args.algorithms),
        "algorithm_labels": {algorithm: ALGORITHM_LABELS[algorithm] for algorithm in args.algorithms},
        "comparisons": [list(pair) for pair in comparisons],
        "seeds": seeds,
        "engine": str(args.engine),
        "pop_size": int(args.pop_size),
        "max_evaluations": int(args.max_evaluations),
        "n_var": int(args.n_var),
        "n_obj": int(args.n_obj),
        "workers": int(workers),
        "alpha": float(args.alpha),
        "tie_tol": float(args.tie_tol),
        "output_dir": str(output_dir),
        "campaign": "farthest_gces_difficulty_ab_zcat_3obj_full",
        "expected_total_runs": int(expected_total_runs),
        "report_title": "Farthest + GCES 3-objective ZCAT difficulty robustness campaign",
        "report_note": (
            "This report extends the canonical 3-objective farthest/GCES survival-only campaign with official ZCAT "
            "difficulty controls. All algorithms keep the same NSGA-II host, mating, variation, and non-dominated "
            "sorting semantics. Only split-front environmental selection differs across algorithms."
        ),
    }

    print("Farthest + GCES 3-objective ZCAT difficulty robustness campaign")
    print(json.dumps(run_config, indent=2))
    print(f"Using {workers} worker(s)")

    records = run_campaign(
        cells=cells,
        problems=list(args.problems),
        algorithms=list(args.algorithms),
        seeds=seeds,
        engine=str(args.engine),
        pop_size=int(args.pop_size),
        max_evaluations=int(args.max_evaluations),
        n_var=int(args.n_var),
        n_obj=int(args.n_obj),
        workers=workers,
        show_progress=True,
    )

    write_csv(output_dir / "raw_results.csv", [asdict(record) for record in records])
    write_run_config(output_dir / "run_config.json", run_config)
    write_analysis_artifacts(
        output_dir=output_dir,
        records=records,
        run_config=run_config,
        alpha=float(args.alpha),
        tie_tol=float(args.tie_tol),
    )

    print(f"\nWrote campaign outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
