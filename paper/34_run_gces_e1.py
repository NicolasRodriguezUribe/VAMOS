"""
Minimal GCES vs NSGA-II benchmark harness for the E2 2-objective ZCAT sweep.

This script is intentionally narrow:
- VAMOS only
- algorithms: nsgaii, gces
- problems: configurable ZCAT subset, defaulting to the full 2-objective ZCAT suite
- objectives: 2
- indicators: normalized HV and IGD+

It reuses:
- `vamos.optimize(...)` for execution
- `paper.benchmark_utils` for fixed-reference HV / IGD+

Outputs are written under `experiments/gces_e2/` by default:
- `raw_results.csv`
- `summary.csv`
- `comparison.csv`
- `run_config.json`
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from vamos import optimize
from vamos.engine.algorithm.config import NSGAIIConfig

try:
    from .benchmark_utils import compute_hv, compute_igd_plus
except ImportError:
    from benchmark_utils import compute_hv, compute_igd_plus


DEFAULT_PROBLEMS = [f"zcat{i}" for i in range(1, 21)]
DEFAULT_ALGORITHMS = ["nsgaii", "gces"]
DEFAULT_POP_SIZE = 100
DEFAULT_EVALS = 25_000
DEFAULT_SEEDS = 21
DEFAULT_N_VAR = 30
DEFAULT_N_OBJ = 2
DEFAULT_ENGINE = "numpy"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "experiments" / "gces_e2"


@dataclass(frozen=True)
class RunRecord:
    problem: str
    algorithm: str
    seed: int
    engine: str
    pop_size: int
    max_evaluations: int
    runtime_seconds: float
    n_solutions: int
    hypervolume: float
    igd_plus: float


def _float_or_zero(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(stdev(values))


def _median_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(median(values))


def _win_tie_loss(
    gces_values: list[float],
    nsgaii_values: list[float],
    *,
    higher_is_better: bool,
    tie_tol: float,
) -> tuple[int, int, int]:
    wins = 0
    ties = 0
    losses = 0
    for gces_val, nsgaii_val in zip(gces_values, nsgaii_values, strict=False):
        diff = float(gces_val - nsgaii_val)
        if abs(diff) <= tie_tol:
            ties += 1
        elif (diff > 0.0 and higher_is_better) or (diff < 0.0 and not higher_is_better):
            wins += 1
        else:
            losses += 1
    return wins, ties, losses


def _paired_wilcoxon_pvalue(
    gces_values: list[float],
    nsgaii_values: list[float],
    *,
    higher_is_better: bool,
) -> float | None:
    try:
        from scipy import stats as spstats  # type: ignore[import-untyped]
    except Exception:
        return None

    if not gces_values or not nsgaii_values or len(gces_values) != len(nsgaii_values):
        return None

    x = list(gces_values)
    y = list(nsgaii_values)
    if not higher_is_better:
        x = [-value for value in x]
        y = [-value for value in y]
    try:
        _stat, p_value = spstats.wilcoxon(x, y, zero_method="pratt", alternative="two-sided")
    except ValueError:
        return None
    return float(p_value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_config(pop_size: int, n_var: int) -> NSGAIIConfig:
    return (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("polynomial", prob=1.0 / n_var, eta=20.0)
        .selection("tournament")
        .build()
    )


def _run_once(
    *,
    problem: str,
    algorithm: str,
    seed: int,
    engine: str,
    pop_size: int,
    max_evaluations: int,
    n_var: int,
    n_obj: int,
    algorithm_config: NSGAIIConfig,
) -> RunRecord:
    start = time.perf_counter()
    result = optimize(
        problem,
        algorithm=algorithm,
        algorithm_config=algorithm_config,
        max_evaluations=max_evaluations,
        seed=seed,
        engine=engine,
        n_var=n_var,
        n_obj=n_obj,
    )
    elapsed = time.perf_counter() - start
    F = result.F
    return RunRecord(
        problem=problem,
        algorithm=algorithm,
        seed=seed,
        engine=engine,
        pop_size=pop_size,
        max_evaluations=max_evaluations,
        runtime_seconds=float(elapsed),
        n_solutions=int(F.shape[0]),
        hypervolume=float(compute_hv(F, problem)),
        igd_plus=float(compute_igd_plus(F, problem)),
    )


def _summaries(
    records: list[RunRecord],
    *,
    add_wilcoxon: bool,
    alpha: float,
    tie_tol: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    by_problem_seed_algo: dict[tuple[str, int, str], RunRecord] = {}

    for record in records:
        grouped[(record.problem, record.algorithm)].append(record)
        by_problem_seed_algo[(record.problem, record.seed, record.algorithm)] = record

    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    problems = sorted({record.problem for record in records})
    algorithms = sorted({record.algorithm for record in records})

    for problem in problems:
        algo_rows: dict[str, list[RunRecord]] = {algo: grouped[(problem, algo)] for algo in algorithms}
        for algorithm, runs in algo_rows.items():
            hv_values = [r.hypervolume for r in runs]
            igd_values = [r.igd_plus for r in runs]
            rt_values = [r.runtime_seconds for r in runs]
            hv_mean, hv_std = _float_or_zero(hv_values)
            igd_mean, igd_std = _float_or_zero(igd_values)
            rt_mean, rt_std = _float_or_zero(rt_values)
            summary_rows.append(
                {
                    "problem": problem,
                    "algorithm": algorithm,
                    "runs": len(runs),
                    "hv_mean": hv_mean,
                    "hv_std": hv_std,
                    "hv_median": _median_or_zero(hv_values),
                    "igd_plus_mean": igd_mean,
                    "igd_plus_std": igd_std,
                    "igd_plus_median": _median_or_zero(igd_values),
                    "runtime_seconds_mean": rt_mean,
                    "runtime_seconds_std": rt_std,
                    "runtime_seconds_median": _median_or_zero(rt_values),
                }
            )

        if "nsgaii" not in algo_rows or "gces" not in algo_rows:
            continue

        nsgaii_runs = {r.seed: r for r in algo_rows["nsgaii"]}
        gces_runs = {r.seed: r for r in algo_rows["gces"]}
        common_seeds = sorted(set(nsgaii_runs) & set(gces_runs))
        gces_hv_values = [gces_runs[seed].hypervolume for seed in common_seeds]
        nsgaii_hv_values = [nsgaii_runs[seed].hypervolume for seed in common_seeds]
        gces_igd_values = [gces_runs[seed].igd_plus for seed in common_seeds]
        nsgaii_igd_values = [nsgaii_runs[seed].igd_plus for seed in common_seeds]

        gces_hv_wins, gces_hv_ties, gces_hv_losses = _win_tie_loss(
            gces_hv_values,
            nsgaii_hv_values,
            higher_is_better=True,
            tie_tol=tie_tol,
        )
        gces_igd_wins, gces_igd_ties, gces_igd_losses = _win_tie_loss(
            gces_igd_values,
            nsgaii_igd_values,
            higher_is_better=False,
            tie_tol=tie_tol,
        )
        both_wins = sum(
            1
            for seed in common_seeds
            if gces_runs[seed].hypervolume > nsgaii_runs[seed].hypervolume and gces_runs[seed].igd_plus < nsgaii_runs[seed].igd_plus
        )

        nsgaii_hv_mean = mean(r.hypervolume for r in algo_rows["nsgaii"])
        gces_hv_mean = mean(r.hypervolume for r in algo_rows["gces"])
        nsgaii_igd_mean = mean(r.igd_plus for r in algo_rows["nsgaii"])
        gces_igd_mean = mean(r.igd_plus for r in algo_rows["gces"])
        hv_p_value = _paired_wilcoxon_pvalue(gces_hv_values, nsgaii_hv_values, higher_is_better=True) if add_wilcoxon else None
        igd_p_value = _paired_wilcoxon_pvalue(gces_igd_values, nsgaii_igd_values, higher_is_better=False) if add_wilcoxon else None

        comparison_rows.append(
            {
                "problem": problem,
                "n_common_seeds": len(common_seeds),
                "nsgaii_hv_mean": float(nsgaii_hv_mean),
                "nsgaii_hv_median": _median_or_zero([r.hypervolume for r in algo_rows["nsgaii"]]),
                "gces_hv_mean": float(gces_hv_mean),
                "gces_hv_median": _median_or_zero([r.hypervolume for r in algo_rows["gces"]]),
                "delta_hv_gces_minus_nsgaii": float(gces_hv_mean - nsgaii_hv_mean),
                "delta_hv_median_gces_minus_nsgaii": _median_or_zero([r.hypervolume for r in algo_rows["gces"]])
                - _median_or_zero([r.hypervolume for r in algo_rows["nsgaii"]]),
                "gces_hv_seed_wins": int(gces_hv_wins),
                "gces_hv_seed_ties": int(gces_hv_ties),
                "gces_hv_seed_losses": int(gces_hv_losses),
                "hv_wilcoxon_p_value": hv_p_value,
                "hv_wilcoxon_significant_alpha": bool(hv_p_value is not None and hv_p_value < alpha),
                "nsgaii_igd_plus_mean": float(nsgaii_igd_mean),
                "nsgaii_igd_plus_median": _median_or_zero([r.igd_plus for r in algo_rows["nsgaii"]]),
                "gces_igd_plus_mean": float(gces_igd_mean),
                "gces_igd_plus_median": _median_or_zero([r.igd_plus for r in algo_rows["gces"]]),
                "delta_igd_plus_gces_minus_nsgaii": float(gces_igd_mean - nsgaii_igd_mean),
                "delta_igd_plus_median_gces_minus_nsgaii": _median_or_zero([r.igd_plus for r in algo_rows["gces"]])
                - _median_or_zero([r.igd_plus for r in algo_rows["nsgaii"]]),
                "gces_igd_plus_seed_wins": int(gces_igd_wins),
                "gces_igd_plus_seed_ties": int(gces_igd_ties),
                "gces_igd_plus_seed_losses": int(gces_igd_losses),
                "igd_plus_wilcoxon_p_value": igd_p_value,
                "igd_plus_wilcoxon_significant_alpha": bool(igd_p_value is not None and igd_p_value < alpha),
                "gces_both_metric_seed_wins": int(both_wins),
            }
        )

    return summary_rows, comparison_rows


def _print_console_summary(summary_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> None:
    print("\nPer-algorithm summary")
    for row in summary_rows:
        print(
            f"  {row['problem']:>6} {row['algorithm']:>6} | "
            f"HV {row['hv_mean']:.4f} +/- {row['hv_std']:.4f} (med {row['hv_median']:.4f}) | "
            f"IGD+ {row['igd_plus_mean']:.4f} +/- {row['igd_plus_std']:.4f} (med {row['igd_plus_median']:.4f}) | "
            f"time {row['runtime_seconds_mean']:.2f}s"
        )

    print("\nProblem-level comparison")
    for row in comparison_rows:
        print(
            f"  {row['problem']:>6} | "
            f"dHV={row['delta_hv_gces_minus_nsgaii']:+.4f} | "
            f"dIGD+={row['delta_igd_plus_gces_minus_nsgaii']:+.4f} | "
            f"GCES HV W/T/L {row['gces_hv_seed_wins']}/{row['gces_hv_seed_ties']}/{row['gces_hv_seed_losses']} | "
            f"GCES IGD+ W/T/L {row['gces_igd_plus_seed_wins']}/{row['gces_igd_plus_seed_ties']}/{row['gces_igd_plus_seed_losses']}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VAMOS-only GCES vs NSGA-II E2 benchmark on the 2-objective ZCAT suite.")
    parser.add_argument("--problems", nargs="+", default=DEFAULT_PROBLEMS, help="Problem keys to run.")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="Number of seeds to run, starting from 0.")
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE, help="Population size.")
    parser.add_argument("--max-evaluations", type=int, default=DEFAULT_EVALS, help="Evaluation budget per run.")
    parser.add_argument("--n-var", type=int, default=DEFAULT_N_VAR, help="Number of decision variables.")
    parser.add_argument("--n-obj", type=int, default=DEFAULT_N_OBJ, help="Number of objectives.")
    parser.add_argument("--engine", default=DEFAULT_ENGINE, help="VAMOS engine/backend.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where CSV outputs are written.")
    parser.add_argument("--wilcoxon", action="store_true", help="Add paired Wilcoxon signed-rank p-values per problem/metric.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level used for Wilcoxon flags.")
    parser.add_argument("--tie-tol", type=float, default=1e-12, help="Absolute tolerance used to count seed-level ties.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    seeds = list(range(int(args.seeds)))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config(pop_size=int(args.pop_size), n_var=int(args.n_var))
    records: list[RunRecord] = []

    print("GCES E2 benchmark")
    print(f"  Problems: {args.problems}")
    print(f"  Algorithms: {DEFAULT_ALGORITHMS}")
    print(f"  Seeds: {seeds}")
    print(f"  Engine: {args.engine}")
    print(f"  pop_size={args.pop_size}, max_evaluations={args.max_evaluations}, n_var={args.n_var}, n_obj={args.n_obj}")

    for problem in args.problems:
        print(f"\nProblem {problem}")
        for algorithm in DEFAULT_ALGORITHMS:
            for seed in seeds:
                record = _run_once(
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
                    f"  {algorithm:>6} seed={seed:>2} | "
                    f"HV={record.hypervolume:.4f} | "
                    f"IGD+={record.igd_plus:.4f} | "
                    f"time={record.runtime_seconds:.2f}s"
                )

    raw_rows = [asdict(record) for record in records]
    summary_rows, comparison_rows = _summaries(
        records,
        add_wilcoxon=bool(args.wilcoxon),
        alpha=float(args.alpha),
        tie_tol=float(args.tie_tol),
    )

    _write_csv(output_dir / "raw_results.csv", raw_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "comparison.csv", comparison_rows)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "problems": list(args.problems),
                "algorithms": DEFAULT_ALGORITHMS,
                "seeds": seeds,
                "engine": args.engine,
                "pop_size": args.pop_size,
                "max_evaluations": args.max_evaluations,
                "n_var": args.n_var,
                "n_obj": args.n_obj,
                "wilcoxon": args.wilcoxon,
                "alpha": args.alpha,
                "tie_tol": args.tie_tol,
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _print_console_summary(summary_rows, comparison_rows)
    print(f"\nWrote results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
