"""Run the full 2-objective ZCAT campaign including nsga2_farthest."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from gces_ablation_zcat_2obj_common import (
    ALGORITHM_LABELS,
    DEFAULT_ENGINE,
    DEFAULT_EVALS,
    DEFAULT_N_OBJ,
    DEFAULT_N_VAR,
    DEFAULT_POP_SIZE,
    DEFAULT_PROBLEMS,
    DEFAULT_SEEDS,
    build_config,
    run_once,
    write_analysis_artifacts,
    write_csv,
    write_run_config,
)


DEFAULT_ALGORITHMS = ["nsgaii", "nsga2_farthest", "gces_nocomp", "gces_nogeo", "gces"]
DEFAULT_COMPARISONS = [
    ("nsga2_farthest", "nsgaii"),
    ("nsga2_farthest", "gces"),
    ("nsga2_farthest", "gces_nocomp"),
    ("nsga2_farthest", "gces_nogeo"),
]
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "experiments" / "nsga2_farthest_zcat_2obj_full"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full 2-objective ZCAT nsga2_farthest campaign.")
    parser.add_argument("--problems", nargs="+", default=DEFAULT_PROBLEMS, help="Problem keys to run.")
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS, help="Algorithm IDs to run.")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="Number of seeds to run, starting from 0.")
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE, help="Population size.")
    parser.add_argument("--max-evaluations", type=int, default=DEFAULT_EVALS, help="Evaluation budget per run.")
    parser.add_argument("--n-var", type=int, default=DEFAULT_N_VAR, help="Number of decision variables.")
    parser.add_argument("--n-obj", type=int, default=DEFAULT_N_OBJ, help="Number of objectives.")
    parser.add_argument("--engine", default=DEFAULT_ENGINE, help="VAMOS engine/backend.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where outputs are written.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level used for Holm-corrected Wilcoxon flags.")
    parser.add_argument("--tie-tol", type=float, default=1e-12, help="Absolute tolerance used to count paired ties.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    seeds = list(range(int(args.seeds)))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(pop_size=int(args.pop_size), n_var=int(args.n_var))
    comparisons = [tuple(pair) for pair in DEFAULT_COMPARISONS if pair[0] in args.algorithms and pair[1] in args.algorithms]
    records = []

    run_config = {
        "problems": list(args.problems),
        "algorithms": list(args.algorithms),
        "algorithm_labels": {algorithm: ALGORITHM_LABELS[algorithm] for algorithm in args.algorithms},
        "comparisons": [list(pair) for pair in comparisons],
        "seeds": seeds,
        "engine": args.engine,
        "pop_size": args.pop_size,
        "max_evaluations": args.max_evaluations,
        "n_var": args.n_var,
        "n_obj": args.n_obj,
        "alpha": args.alpha,
        "tie_tol": args.tie_tol,
        "output_dir": str(output_dir),
        "campaign": "nsga2_farthest_zcat_2obj_full",
        "report_title": "NSGA-II farthest 2-objective ZCAT survival-only campaign",
        "report_note": (
            "This report describes a survival-only campaign. All algorithms in this campaign reuse the "
            "NSGA-II host, mating, variation, and non-dominated sorting. Only split-front environmental "
            "selection differs across nsga2_farthest and the GCES-family variants."
        ),
    }

    print("NSGA-II farthest full ZCAT 2-objective campaign")
    print(json.dumps(run_config, indent=2))

    for problem in args.problems:
        print(f"\nProblem {problem}")
        for algorithm in args.algorithms:
            for seed in seeds:
                record = run_once(
                    problem=problem,
                    algorithm=algorithm,
                    seed=seed,
                    engine=str(args.engine),
                    pop_size=int(args.pop_size),
                    max_evaluations=int(args.max_evaluations),
                    n_var=int(args.n_var),
                    n_obj=int(args.n_obj),
                    algorithm_config=config,
                )
                records.append(record)
                print(
                    f"  {ALGORITHM_LABELS[algorithm]:>14} seed={seed:>2} | "
                    f"HV={record.hypervolume:.4f} | "
                    f"IGD+={record.igd_plus:.4f} | "
                    f"time={record.runtime_seconds:.2f}s"
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
