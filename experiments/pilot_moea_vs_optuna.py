"""
Pilot experiment: MOEATuner vs Optuna (TPE) on NSGA-II.

Problem suite: ZDT1, ZDT2, ZDT4 (2-obj, real-valued)
Config budgets: 50, 200
Seeds: 5 independent tuning runs per cell
MOEA eval budget: 50,000 function evaluations per config evaluation

Total: 3 problems × 2 budgets × 5 seeds × 2 tuners = 60 tuning runs
"""

from __future__ import annotations

import csv
import logging
import os
import sys
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
    TuningTask,
    build_nsgaii_config_space,
)
from vamos.engine.tuning.racing.eval_types import EvalFn
from vamos.engine.tuning.racing.tuning_task import EvalContext
from vamos.experiment.cli.tune import make_evaluator

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------
PROBLEMS = ["zdt1", "zdt2", "zdt4"]
BUDGETS = [50, 200]
SEEDS = list(range(5))
N_VAR = 30
N_OBJ = 2
EVAL_BUDGET = 50_000  # function evaluations per config run
REF_POINT = "1.1,1.1"
ALGORITHM = "nsgaii"
OUTPUT_CSV = Path(__file__).parent / "pilot_moea_vs_optuna.csv"

# Tuning instances: 3 seeds × 1 instance per problem (eval_fn picks problem via ctx.instance.name)
EVAL_SEEDS = [0, 1, 2]


def _build_eval_fn(problem: str) -> EvalFn:
    """Build the evaluation function for a given problem."""
    return make_evaluator(
        problem_key=problem,
        n_var=N_VAR,
        n_obj=N_OBJ,
        algorithm_name=ALGORITHM,
        fixed_pop_size=100,
        ref_point_str=REF_POINT,
        warm_start=False,
        runtime_penalty=0.0,
        failure_score=0.0,
    )


def _build_task(param_space: Any, problem: str) -> TuningTask:
    """Build a TuningTask for a single problem."""
    instances = [Instance(name=problem, n_var=N_VAR, kwargs={})]
    return TuningTask(
        name=f"pilot_{problem}_{ALGORITHM}",
        param_space=param_space,
        instances=instances,
        seeds=EVAL_SEEDS,
        aggregator=lambda scores: float(np.mean(scores)),
        budget_per_run=EVAL_BUDGET,
        maximize=True,
    )


def _run_optuna(
    task: TuningTask,
    eval_fn: EvalFn,
    budget: int,
    seed: int,
) -> tuple[dict[str, Any], float, int]:
    """Run Optuna TPE tuner. Returns (best_config, best_hv, n_trials)."""
    tuner = ModelBasedTuner(
        task=task,
        max_trials=budget,
        backend="optuna",
        seed=seed,
        n_jobs=N_JOBS,
        optuna_sampler="tpe",
    )
    best_config, history = tuner.run(eval_fn, verbose=False)
    finite = [t.score for t in history if np.isfinite(t.score)]
    best_hv = max(finite) if finite else 0.0
    return best_config, best_hv, len(history)


def _run_moea_tuner(
    config_space: AlgorithmConfigSpace,
    task: TuningTask,
    eval_fn: EvalFn,
    budget: int,
    seed: int,
) -> tuple[dict[str, Any], float, int]:
    """Run MOEATuner. Returns (best_config, best_hv, n_trials)."""
    cfg = MOEATunerConfig(
        max_experiments=budget,
        seed=seed,
        n_jobs=N_JOBS,
        verbose=False,
        use_knowledge_base=False,
    )
    tuner = MOEATuner(
        config_space=config_space,
        task=task,
        config=cfg,
    )
    best_config, history = tuner.run(eval_fn, verbose=False)
    finite = [t.score for t in history if np.isfinite(t.score)]
    best_hv = max(finite) if finite else 0.0
    return best_config, best_hv, len(history)


def _load_completed(csv_path: Path) -> set[tuple[str, int, int, str]]:
    """Load already-completed cells from CSV for resume support."""
    completed = set()
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["problem"], int(row["budget"]), int(row["seed"]), row["tuner"])
                completed.add(key)
    return completed


def _append_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Append a single row to the CSV, creating header if needed."""
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem", "budget", "seed", "tuner", "best_hv", "wall_seconds", "n_trials"])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_experiment() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    # Build config space and param space once
    config_space = build_nsgaii_config_space()
    assert isinstance(config_space, AlgorithmConfigSpace)
    param_space = config_space.to_param_space()

    completed = _load_completed(OUTPUT_CSV)
    total_cells = len(PROBLEMS) * len(BUDGETS) * len(SEEDS) * 2
    done_count = len(completed)

    print(f"Pilot: MOEATuner vs Optuna on NSGA-II")
    print(f"Problems: {PROBLEMS}, Budgets: {BUDGETS}, Seeds: {SEEDS}")
    print(f"Eval budget: {EVAL_BUDGET} FEs, Ref point: {REF_POINT}")
    print(f"Already completed: {done_count}/{total_cells}")
    print("-" * 70)

    for problem in PROBLEMS:
        eval_fn = _build_eval_fn(problem)
        task_optuna = _build_task(param_space, problem)
        task_moea = _build_task(param_space, problem)

        for budget in BUDGETS:
            for seed in SEEDS:
                for tuner_name in ["moea_tuner", "optuna"]:
                    key = (problem, budget, seed, tuner_name)
                    if key in completed:
                        continue

                    print(f"  {problem} | budget={budget} | seed={seed} | {tuner_name} ... ", end="", flush=True)
                    t0 = time.perf_counter()

                    try:
                        if tuner_name == "moea_tuner":
                            _, best_hv, n_trials = _run_moea_tuner(
                                config_space, task_moea, eval_fn, budget, seed,
                            )
                        else:
                            _, best_hv, n_trials = _run_optuna(
                                task_optuna, eval_fn, budget, seed,
                            )
                        wall = time.perf_counter() - t0
                        print(f"HV={best_hv:.4f} ({wall:.1f}s, {n_trials} trials)")
                    except Exception as e:
                        wall = time.perf_counter() - t0
                        best_hv = 0.0
                        n_trials = 0
                        print(f"FAILED: {e} ({wall:.1f}s)")

                    _append_row(OUTPUT_CSV, {
                        "problem": problem,
                        "budget": budget,
                        "seed": seed,
                        "tuner": tuner_name,
                        "best_hv": f"{best_hv:.6f}",
                        "wall_seconds": f"{wall:.2f}",
                        "n_trials": n_trials,
                    })
                    done_count += 1

    print("-" * 70)
    print(f"Done. Results saved to {OUTPUT_CSV}")
    _print_summary()


def _print_summary() -> None:
    """Print a summary table from the CSV."""
    if not OUTPUT_CSV.exists():
        print("No results file found.")
        return

    import pandas as pd

    df = pd.read_csv(OUTPUT_CSV)
    print("\n=== Summary: Mean HV (std) by tuner × budget ===")
    pivot = df.groupby(["tuner", "budget"])["best_hv"].agg(["mean", "std", "count"])
    print(pivot.to_string())

    print("\n=== Per-problem breakdown ===")
    pivot2 = df.groupby(["problem", "tuner", "budget"])["best_hv"].agg(["mean", "std"])
    print(pivot2.to_string())

    # Paired comparison per (problem, budget)
    print("\n=== Paired comparison (MOEATuner - Optuna) ===")
    for problem in df["problem"].unique():
        for budget in df["budget"].unique():
            moea = df[(df["problem"] == problem) & (df["budget"] == budget) & (df["tuner"] == "moea_tuner")]["best_hv"].values
            opt = df[(df["problem"] == problem) & (df["budget"] == budget) & (df["tuner"] == "optuna")]["best_hv"].values
            if len(moea) > 0 and len(opt) > 0 and len(moea) == len(opt):
                diff = moea - opt
                print(f"  {problem} budget={budget}: diff={np.mean(diff):+.4f} (±{np.std(diff):.4f})")

    # Wall-clock comparison
    print("\n=== Wall-clock: Mean seconds by tuner × budget ===")
    wall_pivot = df.groupby(["tuner", "budget"])["wall_seconds"].agg(["mean", "std"])
    print(wall_pivot.to_string())


if __name__ == "__main__":
    run_experiment()
