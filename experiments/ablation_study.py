"""
Ablation study: quantify the contribution of each MOEATuner component.

6 Variants:
  1. Full MOEATuner        (hierarchical phases A→B→C)
  2. No Phase A            (skip structural, start at B)
  3. No Phase B            (skip operator selection, jump A→C)
  4. No Cheap Evals        (all phases use full budget)
  5. Random Phase Order    (shuffle role groups across phases)
  6. Flat Racing           (standard RacingTuner, no phases)

Design:
  Algorithm: NSGA-II
  Problems: ZDT1, DTLZ2, WFG1, WFG4 (diverse Pareto front shapes)
  Budget: 200 config evaluations per tuning run
  Seeds: 31 independent runs
  Metric: HV (hypervolume)

Usage:
  python experiments/ablation_study.py [--smoke] [--seeds 31] [--budget 200]
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
    RacingTuner,
    Scenario,
    TuningTask,
    build_nsgaii_config_space,
)
from vamos.engine.tuning.moea_tuner import PHASE_A_ROLES, PHASE_B_ROLES, PHASE_C_ROLES
from vamos.engine.tuning.racing.eval_types import EvalFn
from vamos.engine.tuning.racing.tuning_task import EvalContext
from vamos.experiment.cli.tune import make_evaluator

OUTPUT_CSV = Path(__file__).parent / "ablation_study.csv"

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------
PROBLEMS = {
    "zdt1":  {"n_var": 30, "n_obj": 2, "ref": "1.1,1.1"},
    "dtlz2": {"n_var": 12, "n_obj": 2, "ref": "1.1,1.1"},
    "wfg1":  {"n_var": 10, "n_obj": 2, "ref": "3.0,5.0"},
    "wfg4":  {"n_var": 10, "n_obj": 2, "ref": "3.0,5.0"},
}
ALGORITHM = "nsgaii"
EVAL_BUDGET = 50_000
EVAL_SEEDS = [0, 1, 2]


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

def _build_eval_fn(problem: str, info: dict) -> EvalFn:
    return make_evaluator(
        problem_key=problem,
        n_var=info["n_var"],
        n_obj=info["n_obj"],
        algorithm_name=ALGORITHM,
        fixed_pop_size=100,
        ref_point_str=info["ref"],
        warm_start=False,
        runtime_penalty=0.0,
        failure_score=0.0,
    )


def _build_task(param_space: Any, problem: str, info: dict) -> TuningTask:
    return TuningTask(
        name=f"ablation_{problem}",
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


def run_full_moeatuner(
    config_space: AlgorithmConfigSpace,
    task: TuningTask,
    eval_fn: EvalFn,
    budget: int,
    seed: int,
) -> tuple[float, int]:
    """Variant 1: Full MOEATuner (hierarchical A→B→C)."""
    cfg = MOEATunerConfig(
        max_experiments=budget,
        seed=seed,
        n_jobs=N_JOBS,
        verbose=False,
        use_knowledge_base=False,
    )
    tuner = MOEATuner(config_space=config_space, task=task, config=cfg)
    _, history = tuner.run(eval_fn, verbose=False)
    return _extract_best_hv(history), len(history)


def run_no_phase_a(
    config_space: AlgorithmConfigSpace,
    task: TuningTask,
    eval_fn: EvalFn,
    budget: int,
    seed: int,
) -> tuple[float, int]:
    """Variant 2: Skip Phase A (no structural tuning)."""
    cfg = MOEATunerConfig(
        max_experiments=budget,
        budget_split=(0.0, 0.5, 0.5),
        seed=seed,
        n_jobs=N_JOBS,
        verbose=False,
        use_knowledge_base=False,
    )
    tuner = MOEATuner(config_space=config_space, task=task, config=cfg)
    _, history = tuner.run(eval_fn, verbose=False)
    return _extract_best_hv(history), len(history)


def run_no_phase_b(
    config_space: AlgorithmConfigSpace,
    task: TuningTask,
    eval_fn: EvalFn,
    budget: int,
    seed: int,
) -> tuple[float, int]:
    """Variant 3: Skip Phase B (no operator selection)."""
    cfg = MOEATunerConfig(
        max_experiments=budget,
        budget_split=(0.3, 0.0, 0.7),
        seed=seed,
        n_jobs=N_JOBS,
        verbose=False,
        use_knowledge_base=False,
    )
    tuner = MOEATuner(config_space=config_space, task=task, config=cfg)
    _, history = tuner.run(eval_fn, verbose=False)
    return _extract_best_hv(history), len(history)


def run_no_cheap_evals(
    config_space: AlgorithmConfigSpace,
    task: TuningTask,
    eval_fn: EvalFn,
    budget: int,
    seed: int,
) -> tuple[float, int]:
    """Variant 4: All phases use full evaluation budget (no cheap Phase A)."""
    full_budget = task.budget_per_run
    cfg = MOEATunerConfig(
        max_experiments=budget,
        budget_per_run_phases=(full_budget, full_budget, full_budget),
        seed=seed,
        n_jobs=N_JOBS,
        verbose=False,
        use_knowledge_base=False,
    )
    tuner = MOEATuner(config_space=config_space, task=task, config=cfg)
    _, history = tuner.run(eval_fn, verbose=False)
    return _extract_best_hv(history), len(history)


def run_random_phase_order(
    config_space: AlgorithmConfigSpace,
    task: TuningTask,
    eval_fn: EvalFn,
    budget: int,
    seed: int,
) -> tuple[float, int]:
    """Variant 5: Shuffle role groups across phases (random order)."""
    rng = np.random.default_rng(seed)
    roles = [PHASE_A_ROLES, PHASE_B_ROLES, PHASE_C_ROLES]
    order = list(range(3))
    rng.shuffle(order)
    # We can't easily change MOEATuner phase order externally, so we
    # simulate by running 3 single-phase MOEATuners in shuffled order
    # with frozen params carried forward.
    full_space = config_space.to_param_space()
    role_groups = config_space.group_by_role()
    phase_budgets = [
        max(1, int(budget * 0.3)),
        max(1, int(budget * 0.35)),
        budget - max(1, int(budget * 0.3)) - max(1, int(budget * 0.35)),
    ]
    frozen: dict[str, Any] = {}
    all_history: list = []

    # Create a temporary MOEATuner just to access _run_phase
    cfg = MOEATunerConfig(
        max_experiments=budget, seed=seed, n_jobs=N_JOBS, verbose=False,
        use_knowledge_base=False,
    )
    tuner = MOEATuner(config_space=config_space, task=task, config=cfg)

    for idx in order:
        active_roles = roles[idx]
        phase_budget = phase_budgets[idx]
        result = tuner._run_phase(
            phase_name=f"Shuffled-{idx}",
            active_roles=active_roles,
            frozen_params=dict(frozen),
            full_param_space=full_space,
            role_groups=role_groups,
            eval_fn=eval_fn,
            max_experiments=phase_budget,
            budget_per_run=EVAL_BUDGET,
            verbose=False,
        )
        frozen.update(result.best_config)
        all_history.extend(result.history)

    hv = _extract_best_hv(all_history) if all_history else 0.0
    return hv, len(all_history)


def run_flat_racing(
    config_space: AlgorithmConfigSpace,
    task: TuningTask,
    eval_fn: EvalFn,
    budget: int,
    seed: int,
) -> tuple[float, int]:
    """Variant 6: Standard flat racing (no hierarchical phases)."""
    scenario = Scenario(max_experiments=budget, n_jobs=N_JOBS)
    tuner = RacingTuner(
        task=task,
        scenario=scenario,
        seed=seed,
        max_initial_configs=min(20, budget),
    )
    _, history = tuner.run(eval_fn, verbose=False)
    return _extract_best_hv(history), len(history)


VARIANTS = {
    "full_moeatuner": run_full_moeatuner,
    "no_phase_a": run_no_phase_a,
    "no_phase_b": run_no_phase_b,
    "no_cheap_evals": run_no_cheap_evals,
    "random_phase_order": run_random_phase_order,
    "flat_racing": run_flat_racing,
}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

FIELDNAMES = ["problem", "variant", "seed", "best_hv", "wall_seconds", "n_trials"]


def _load_completed(csv_path: Path) -> set[tuple[str, str, int]]:
    completed = set()
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                completed.add((row["problem"], row["variant"], int(row["seed"])))
    return completed


def _append_row(csv_path: Path, row: dict[str, Any]) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_ablation(budget: int = 200, seeds: list[int] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if seeds is None:
        seeds = list(range(31))

    config_space = build_nsgaii_config_space()
    param_space = config_space.to_param_space()
    completed = _load_completed(OUTPUT_CSV)

    total = len(PROBLEMS) * len(VARIANTS) * len(seeds)
    done = len(completed)
    print(f"Ablation Study: {ALGORITHM}")
    print(f"  Problems: {list(PROBLEMS.keys())}")
    print(f"  Variants: {list(VARIANTS.keys())}")
    print(f"  Budget: {budget} configs, Seeds: {len(seeds)}")
    print(f"  Already completed: {done}/{total}")
    print("-" * 70)

    for problem, info in PROBLEMS.items():
        eval_fn = _build_eval_fn(problem, info)
        task = _build_task(param_space, problem, info)

        for variant_name, variant_fn in VARIANTS.items():
            for seed in seeds:
                key = (problem, variant_name, seed)
                if key in completed:
                    continue

                print(f"  {problem} | {variant_name:20s} | seed={seed:2d} ... ", end="", flush=True)
                t0 = time.perf_counter()

                try:
                    best_hv, n_trials = variant_fn(config_space, task, eval_fn, budget, seed)
                    wall = time.perf_counter() - t0
                    print(f"HV={best_hv:.4f} ({wall:.1f}s, {n_trials} trials)")
                except Exception as e:
                    wall = time.perf_counter() - t0
                    best_hv = 0.0
                    n_trials = 0
                    print(f"FAILED: {e} ({wall:.1f}s)")

                _append_row(OUTPUT_CSV, {
                    "problem": problem,
                    "variant": variant_name,
                    "seed": seed,
                    "best_hv": f"{best_hv:.6f}",
                    "wall_seconds": f"{wall:.2f}",
                    "n_trials": n_trials,
                })
                done += 1

    print("-" * 70)
    print(f"Done. Results saved to {OUTPUT_CSV}")
    _print_summary()


def _print_summary() -> None:
    if not OUTPUT_CSV.exists():
        print("No results file found.")
        return

    import pandas as pd

    df = pd.read_csv(OUTPUT_CSV)
    print("\n=== Mean HV (std) by variant ===")
    agg = df.groupby("variant")["best_hv"].agg(["mean", "std", "count"])
    print(agg.sort_values("mean", ascending=False).to_string())

    print("\n=== Per-problem breakdown ===")
    pivot = df.groupby(["problem", "variant"])["best_hv"].agg(["mean", "std"])
    print(pivot.to_string())

    # Friedman test across variants
    try:
        from scipy.stats import friedmanchisquare

        print("\n=== Friedman Test ===")
        for problem in df["problem"].unique():
            sub = df[df["problem"] == problem]
            groups = []
            variant_names = []
            for v in VARIANTS:
                vals = sub[sub["variant"] == v]["best_hv"].values
                if len(vals) > 0:
                    groups.append(vals)
                    variant_names.append(v)
            if len(groups) >= 3 and all(len(g) == len(groups[0]) for g in groups):
                stat, p = friedmanchisquare(*groups)
                print(f"  {problem}: chi2={stat:.2f}, p={p:.4f}")
            else:
                print(f"  {problem}: insufficient data for Friedman test")
    except ImportError:
        print("(scipy not available for Friedman test)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study for MOEATuner")
    parser.add_argument("--budget", type=int, default=200, help="Config evaluation budget")
    parser.add_argument("--seeds", type=int, default=31, help="Number of seeds")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (2 seeds, 1 problem)")
    args = parser.parse_args()

    if args.smoke:
        # Override for smoke test
        global PROBLEMS
        PROBLEMS = {"zdt1": PROBLEMS["zdt1"]}
        run_ablation(budget=4, seeds=[0, 1])
    else:
        run_ablation(budget=args.budget, seeds=list(range(args.seeds)))


if __name__ == "__main__":
    main()
