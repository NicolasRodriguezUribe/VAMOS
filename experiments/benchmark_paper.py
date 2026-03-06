"""
Full benchmark: MOEATuner vs Optuna (TPE) vs Flat Racing vs Random Search.

Design for IEEE TEC paper:
  Algorithms: NSGA-II (primary), MOEA/D, SMS-EMOA (secondary)
  Problems: ZDT1, ZDT2, ZDT4, DTLZ2, DTLZ3, WFG1, WFG4
  Budgets: 50, 100, 200 config evaluations
  Seeds: 31 independent tuning runs
  Metrics: HV (hypervolume)
  Statistics: Friedman + Nemenyi post-hoc, Vargha-Delaney A12

Usage:
  python experiments/benchmark_paper.py [--smoke] [--algorithm nsgaii] [--seeds 31]
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

N_JOBS = max(1, os.cpu_count() - 1)

from vamos.engine.tuning import (
    AlgorithmConfigSpace,
    Instance,
    MOEATuner,
    MOEATunerConfig,
    ModelBasedTuner,
    RacingTuner,
    RandomSearchTuner,
    Scenario,
    TuningTask,
    build_moead_config_space,
    build_nsgaii_config_space,
    build_smsemoa_config_space,
    vargha_delaney_a12,
    a12_magnitude,
)
from vamos.engine.tuning.racing.eval_types import EvalFn
from vamos.engine.tuning.racing.tuning_task import EvalContext
from vamos.experiment.cli.tune import make_evaluator

OUTPUT_CSV = Path(__file__).parent / "benchmark_tuner_comparison.csv"

# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------
PROBLEMS = {
    "zdt1":  {"n_var": 30, "n_obj": 2, "ref": "1.1,1.1"},
    "zdt2":  {"n_var": 30, "n_obj": 2, "ref": "1.1,1.1"},
    "zdt4":  {"n_var": 10, "n_obj": 2, "ref": "1.1,1.1"},
    "dtlz2": {"n_var": 12, "n_obj": 2, "ref": "1.1,1.1"},
    "dtlz3": {"n_var": 12, "n_obj": 2, "ref": "1.1,1.1"},
    "wfg1":  {"n_var": 10, "n_obj": 2, "ref": "3.0,5.0"},
    "wfg4":  {"n_var": 10, "n_obj": 2, "ref": "3.0,5.0"},
}

ALGORITHM_BUILDERS = {
    "nsgaii": build_nsgaii_config_space,
    "moead": build_moead_config_space,
    "smsemoa": build_smsemoa_config_space,
}

BUDGETS = [50, 100, 200]
EVAL_BUDGET = 50_000
EVAL_SEEDS = [0, 1, 2]

FIELDNAMES = [
    "algorithm", "problem", "budget", "seed", "tuner",
    "best_hv", "wall_seconds", "n_trials",
]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _build_eval_fn(algorithm: str, problem: str, info: dict) -> EvalFn:
    return make_evaluator(
        problem_key=problem,
        n_var=info["n_var"],
        n_obj=info["n_obj"],
        algorithm_name=algorithm,
        fixed_pop_size=100,
        ref_point_str=info["ref"],
        warm_start=False,
        runtime_penalty=0.0,
        failure_score=0.0,
    )


def _build_task(param_space: Any, problem: str, info: dict) -> TuningTask:
    return TuningTask(
        name=f"bench_{problem}",
        param_space=param_space,
        instances=[Instance(name=problem, n_var=info["n_var"], kwargs={})],
        seeds=EVAL_SEEDS,
        aggregator=lambda scores: float(np.mean(scores)),
        budget_per_run=EVAL_BUDGET,
        maximize=True,
    )


def _extract_best_hv(history: list) -> float:
    finite = [t.score for t in history if np.isfinite(t.score)]
    return max(finite) if finite else 0.0


# ---------------------------------------------------------------------------
# Tuner runners
# ---------------------------------------------------------------------------

def run_moea_tuner(cs, task, eval_fn, budget, seed):
    cfg = MOEATunerConfig(
        max_experiments=budget, seed=seed, n_jobs=N_JOBS,
        verbose=False, use_knowledge_base=False,
    )
    tuner = MOEATuner(config_space=cs, task=task, config=cfg)
    _, history = tuner.run(eval_fn, verbose=False)
    return _extract_best_hv(history), len(history)


def run_optuna(cs, task, eval_fn, budget, seed):
    tuner = ModelBasedTuner(
        task=task, max_trials=budget, backend="optuna",
        seed=seed, n_jobs=N_JOBS, optuna_sampler="tpe",
    )
    _, history = tuner.run(eval_fn, verbose=False)
    return _extract_best_hv(history), len(history)


def run_flat_racing(cs, task, eval_fn, budget, seed):
    scenario = Scenario(max_experiments=budget, n_jobs=N_JOBS)
    tuner = RacingTuner(
        task=task, scenario=scenario, seed=seed,
        max_initial_configs=min(20, budget),
    )
    _, history = tuner.run(eval_fn, verbose=False)
    return _extract_best_hv(history), len(history)


def run_random_search(cs, task, eval_fn, budget, seed):
    tuner = RandomSearchTuner(task=task, max_trials=budget, seed=seed)
    _, history = tuner.run(eval_fn, verbose=False)
    return _extract_best_hv(history), len(history)


TUNERS = {
    "moea_tuner": run_moea_tuner,
    "optuna": run_optuna,
    "flat_racing": run_flat_racing,
    "random_search": run_random_search,
}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _load_completed(csv_path: Path) -> set[tuple]:
    completed = set()
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                completed.add((
                    row["algorithm"], row["problem"],
                    int(row["budget"]), int(row["seed"]), row["tuner"],
                ))
    return completed


def _append_row(csv_path: Path, row: dict[str, Any]) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_benchmark(
    algorithm: str = "nsgaii",
    budgets: list[int] | None = None,
    seeds: list[int] | None = None,
    problems: dict | None = None,
) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if budgets is None:
        budgets = BUDGETS
    if seeds is None:
        seeds = list(range(31))
    if problems is None:
        problems = PROBLEMS

    builder = ALGORITHM_BUILDERS[algorithm]
    config_space = builder()
    assert isinstance(config_space, AlgorithmConfigSpace)
    param_space = config_space.to_param_space()

    completed = _load_completed(OUTPUT_CSV)
    total = len(problems) * len(budgets) * len(seeds) * len(TUNERS)
    done = len(completed)

    print(f"Benchmark: {algorithm}")
    print(f"  Problems: {list(problems.keys())}")
    print(f"  Budgets: {budgets}, Seeds: {len(seeds)}")
    print(f"  Tuners: {list(TUNERS.keys())}")
    print(f"  Eval budget: {EVAL_BUDGET} FEs, n_jobs={N_JOBS}")
    print(f"  Already completed: {done}/{total}")
    print("-" * 70)

    for problem, info in problems.items():
        eval_fn = _build_eval_fn(algorithm, problem, info)
        task = _build_task(param_space, problem, info)

        for budget in budgets:
            for seed in seeds:
                for tuner_name, tuner_fn in TUNERS.items():
                    key = (algorithm, problem, budget, seed, tuner_name)
                    if key in completed:
                        continue

                    print(f"  {problem} | B={budget} | s={seed:2d} | {tuner_name:15s} ... ",
                          end="", flush=True)
                    t0 = time.perf_counter()

                    try:
                        best_hv, n_trials = tuner_fn(config_space, task, eval_fn, budget, seed)
                        wall = time.perf_counter() - t0
                        print(f"HV={best_hv:.4f} ({wall:.1f}s, {n_trials}t)")
                    except Exception as e:
                        wall = time.perf_counter() - t0
                        best_hv = 0.0
                        n_trials = 0
                        print(f"FAILED: {e} ({wall:.1f}s)")

                    _append_row(OUTPUT_CSV, {
                        "algorithm": algorithm,
                        "problem": problem,
                        "budget": budget,
                        "seed": seed,
                        "tuner": tuner_name,
                        "best_hv": f"{best_hv:.6f}",
                        "wall_seconds": f"{wall:.2f}",
                        "n_trials": n_trials,
                    })
                    done += 1

    print("-" * 70)
    print(f"Done. Results saved to {OUTPUT_CSV}")
    _print_summary(algorithm)


def _print_summary(algorithm: str) -> None:
    if not OUTPUT_CSV.exists():
        return

    import pandas as pd

    df = pd.read_csv(OUTPUT_CSV)
    df = df[df["algorithm"] == algorithm]

    if df.empty:
        return

    print(f"\n=== Summary: {algorithm} ===")
    print("\n--- Mean HV (std) by tuner x budget ---")
    agg = df.groupby(["tuner", "budget"])["best_hv"].agg(["mean", "std", "count"])
    print(agg.to_string())

    print("\n--- Per-problem: Mean HV by tuner ---")
    pivot = df.groupby(["problem", "tuner"])["best_hv"].agg(["mean", "std"])
    print(pivot.to_string())

    # Vargha-Delaney A12: MOEATuner vs each competitor
    print("\n--- Vargha-Delaney A12: moea_tuner vs competitors ---")
    for budget in sorted(df["budget"].unique()):
        print(f"\n  Budget = {budget}:")
        for problem in sorted(df["problem"].unique()):
            moea_vals = df[
                (df["problem"] == problem) &
                (df["budget"] == budget) &
                (df["tuner"] == "moea_tuner")
            ]["best_hv"].values

            for comp in ["optuna", "flat_racing", "random_search"]:
                comp_vals = df[
                    (df["problem"] == problem) &
                    (df["budget"] == budget) &
                    (df["tuner"] == comp)
                ]["best_hv"].values

                if len(moea_vals) > 0 and len(comp_vals) > 0:
                    a12 = vargha_delaney_a12(np.array(moea_vals), np.array(comp_vals))
                    mag = a12_magnitude(a12)
                    print(f"    {problem:8s} vs {comp:15s}: A12={a12:.3f} ({mag})")

    # Friedman test
    try:
        from scipy.stats import friedmanchisquare
        print("\n--- Friedman Test (across tuners per problem x budget) ---")
        for problem in sorted(df["problem"].unique()):
            for budget in sorted(df["budget"].unique()):
                groups = []
                names = []
                for t in TUNERS:
                    vals = df[
                        (df["problem"] == problem) &
                        (df["budget"] == budget) &
                        (df["tuner"] == t)
                    ]["best_hv"].values
                    if len(vals) > 0:
                        groups.append(vals)
                        names.append(t)
                if len(groups) >= 3 and all(len(g) == len(groups[0]) for g in groups):
                    stat, p = friedmanchisquare(*groups)
                    sig = "*" if p < 0.05 else ""
                    print(f"  {problem:8s} B={budget}: chi2={stat:.2f}, p={p:.4f} {sig}")
    except ImportError:
        pass

    # Wall-clock
    print("\n--- Wall-clock: Mean seconds by tuner x budget ---")
    wall = df.groupby(["tuner", "budget"])["wall_seconds"].agg(["mean", "std"])
    print(wall.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Full benchmark for IEEE TEC paper")
    parser.add_argument("--algorithm", type=str, default="nsgaii",
                        choices=list(ALGORITHM_BUILDERS.keys()))
    parser.add_argument("--seeds", type=int, default=31)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (2 seeds, 1 problem, budget 4)")
    args = parser.parse_args()

    if args.smoke:
        run_benchmark(
            algorithm=args.algorithm,
            budgets=[4],
            seeds=[0, 1],
            problems={"zdt1": PROBLEMS["zdt1"]},
        )
    else:
        run_benchmark(
            algorithm=args.algorithm,
            seeds=list(range(args.seeds)),
        )


if __name__ == "__main__":
    main()
