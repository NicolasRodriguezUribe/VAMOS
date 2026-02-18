"""
WFG 3-Objective Benchmark
=========================
Runs NSGA-II on WFG problems with 3 objectives (m=3, k=4, l=20, n=24) to
demonstrate that the framework scales beyond the 2-objective WFG configuration
used in the main benchmark.

Usage: python paper/24_run_wfg3d_benchmark.py

Environment variables:
  - VAMOS_N_EVALS: evaluations per run (default: 50000)
  - VAMOS_N_SEEDS: number of seeds (default: 30)

Output: experiments/benchmark_paper_wfg3d.csv

Note: Reference fronts for 3-objective WFG need to be generated first.
      Use experiments/scripts/generate_reference_fronts.py if available,
      or provide them in src/vamos/foundation/data/reference_fronts/ as
      wfg1_3d.csv, wfg4_3d.csv, wfg9_3d.csv.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

DESKTOP_DIR = ROOT_DIR.parent
JMETALPY_SRC = DESKTOP_DIR / "jMetalPy" / "src"
PLATYPUS_SRC = DESKTOP_DIR / "Platypus"
for extra_path in (JMETALPY_SRC, PLATYPUS_SRC):
    if extra_path.exists():
        sys.path.insert(0, str(extra_path))

try:
    from .benchmark_utils import compute_hv, compute_igd_plus
except ImportError:
    from benchmark_utils import compute_hv, compute_igd_plus

# =============================================================================
# Configuration
# =============================================================================

N_EVALS = int(os.environ.get("VAMOS_N_EVALS", "50000"))
N_SEEDS = int(os.environ.get("VAMOS_N_SEEDS", "30"))
POP_SIZE = 100
CROSSOVER_PROB = 1.0
CROSSOVER_ETA = 20.0
MUTATION_ETA = 20.0

DATA_DIR = ROOT_DIR / "experiments"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = DATA_DIR / "benchmark_paper_wfg3d.csv"

# WFG 3-objective configuration
N_OBJ = 3
K = 2 * (N_OBJ - 1)  # k = 4 for m=3
L = 20
N_VAR = K + L  # n = 24

# Selected WFG problems for 3-objective study
WFG_PROBLEMS = ["wfg1", "wfg4", "wfg9"]


# =============================================================================
# VAMOS runner
# =============================================================================

def run_vamos(problem_name: str, seed: int) -> dict | None:
    from vamos.foundation.problem.registry import make_problem_selection
    from vamos import optimize
    from vamos.engine.algorithm.config import NSGAIIConfig

    problem = make_problem_selection(problem_name, n_var=N_VAR, n_obj=N_OBJ).instantiate()

    cfg = (
        NSGAIIConfig.builder()
        .pop_size(POP_SIZE)
        .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
        .mutation("pm", prob=1.0 / N_VAR, eta=MUTATION_ETA)
        .selection("tournament")
        .build()
    )

    # Numba warmup
    _ = optimize(problem, algorithm="nsgaii", algorithm_config=cfg,
                 termination=("max_evaluations", min(2000, N_EVALS)),
                 seed=seed, engine="numba")

    start = time.perf_counter()
    result = optimize(problem, algorithm="nsgaii", algorithm_config=cfg,
                      termination=("max_evaluations", N_EVALS),
                      seed=seed, engine="numba")
    elapsed = time.perf_counter() - start

    F = result.F
    n_solutions = result.X.shape[0] if result.X is not None else 0

    # Use 3D problem name for reference front lookup (e.g., wfg1_3d)
    ref_name = f"{problem_name}_3d"
    try:
        hv = compute_hv(F, ref_name) if F is not None else float("nan")
    except FileNotFoundError:
        # Fall back to using the 2D reference front name
        try:
            hv = compute_hv(F, problem_name) if F is not None else float("nan")
        except Exception:
            hv = float("nan")

    try:
        igd_plus = compute_igd_plus(F, ref_name) if F is not None else float("nan")
    except FileNotFoundError:
        try:
            igd_plus = compute_igd_plus(F, problem_name) if F is not None else float("nan")
        except Exception:
            igd_plus = float("nan")

    return {
        "framework": "VAMOS (numba)",
        "problem": f"{problem_name}_3d",
        "algorithm": "NSGA-II",
        "n_evals": N_EVALS,
        "seed": seed,
        "runtime_seconds": elapsed,
        "n_solutions": n_solutions,
        "hypervolume": hv,
        "igd_plus": igd_plus,
    }


# =============================================================================
# pymoo runner
# =============================================================================

def run_pymoo(problem_name: str, seed: int) -> dict | None:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.problems import get_problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.selection.rnd import RandomSelection

    pymoo_problem = get_problem(problem_name, n_var=N_VAR, n_obj=N_OBJ)

    algorithm = NSGA2(
        pop_size=POP_SIZE,
        crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
        mutation=PM(prob=1.0 / N_VAR, eta=MUTATION_ETA),
        selection=RandomSelection(),
    )

    start = time.perf_counter()
    res = minimize(pymoo_problem, algorithm, get_termination("n_eval", N_EVALS),
                   seed=seed, verbose=False)
    elapsed = time.perf_counter() - start

    F = res.F
    n_solutions = res.X.shape[0] if res.X is not None else 0

    ref_name = f"{problem_name}_3d"
    try:
        hv = compute_hv(F, ref_name) if F is not None else float("nan")
    except FileNotFoundError:
        try:
            hv = compute_hv(F, problem_name) if F is not None else float("nan")
        except Exception:
            hv = float("nan")

    try:
        igd_plus = compute_igd_plus(F, ref_name) if F is not None else float("nan")
    except FileNotFoundError:
        try:
            igd_plus = compute_igd_plus(F, problem_name) if F is not None else float("nan")
        except Exception:
            igd_plus = float("nan")

    return {
        "framework": "pymoo",
        "problem": f"{problem_name}_3d",
        "algorithm": "NSGA-II",
        "n_evals": N_EVALS,
        "seed": seed,
        "runtime_seconds": elapsed,
        "n_solutions": n_solutions,
        "hypervolume": hv,
        "igd_plus": igd_plus,
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("  WFG 3-Objective Benchmark")
    print("=" * 60)
    print(f"Problems: {WFG_PROBLEMS} (m={N_OBJ}, k={K}, l={L}, n={N_VAR})")
    print(f"Seeds: {N_SEEDS}")
    print(f"Evaluations: {N_EVALS}")
    print(f"Output: {OUTPUT_CSV}")
    print()

    results: list[dict] = []
    seeds = list(range(N_SEEDS))
    runners = [
        ("VAMOS (numba)", run_vamos),
        ("pymoo", run_pymoo),
    ]

    for problem_name in WFG_PROBLEMS:
        print(f"\n--- {problem_name} (3-obj) ---")
        for seed in seeds:
            for fw_name, run_fn in runners:
                try:
                    entry = run_fn(problem_name, seed)
                    if entry is not None:
                        results.append(entry)
                        rt = entry["runtime_seconds"]
                        hv = entry.get("hypervolume", float("nan"))
                        print(f"  {problem_name} {fw_name} seed={seed}: {rt:.2f}s HV={hv:.4f}")
                except Exception as e:
                    print(f"  {problem_name} {fw_name} seed={seed} FAILED: {e}")

        # Save partial
        if results:
            df = pd.DataFrame(results)
            df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone! Saved {len(results)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
