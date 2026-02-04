"""
VAMOS Paper Scaling Experiment
=============================
Runs scaling experiments to characterize when array-based acceleration helps most.

Design goals:
  - Same evaluation budget and seed count as the main benchmark.
  - Compare VAMOS backends (NumPy vs Numba) under identical NSGA-II settings.
  - Report runtime-per-evaluation and derived speedups in the paper tables.

Usage:
  python paper/03_run_scaling_experiment.py

Environment variables:
  - VAMOS_N_EVALS: evaluations per run (default: 50000)
  - VAMOS_N_SEEDS: number of seeds (default: 30)
  - VAMOS_N_JOBS: joblib workers (default: CPU count - 1)
  - VAMOS_SCALING_ENGINES: comma-separated engines (default: numpy,numba)
  - VAMOS_SCALING_OUTPUT_CSV: output CSV path (default: experiments/scaling_vectorization.csv)

Numba warmup controls (recommended for stable throughput timings):
  - VAMOS_SCALING_NUMBA_WARMUP_EVALS: warmup evaluations before timing a Numba run
      (default: 2000; set 0 to disable)

Population-size scaling controls:
  - VAMOS_SCALING_POP_ENABLED: 1/0 (default: 1)
  - VAMOS_SCALING_POP_SIZES: comma-separated pop sizes (default: 50,100,200,400,800)

Objective-count scaling controls:
  - VAMOS_SCALING_OBJ_ENABLED: 1/0 (default: 1)
  - VAMOS_SCALING_OBJ_COUNTS: comma-separated objective counts (default: 2,3,5,8)
  - VAMOS_SCALING_OBJ_POP_FACTOR: pop size factor for objectives scaling (default: 25)
      pop_size = factor * n_obj

Numba JIT policy experiment (cold vs warm; for timing methodology):
  - VAMOS_SCALING_JIT_ENABLED: 1/0 (default: 1)
  - VAMOS_SCALING_JIT_PROBLEMS: comma-separated problems (default: zdt4,dtlz2,wfg2)
  - VAMOS_SCALING_JIT_POP_SIZE: population size (default: 100)
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from vamos import optimize
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.foundation.problem.registry import make_problem_selection

try:
    from .benchmark_utils import compute_hv
except ImportError:  # pragma: no cover
    from benchmark_utils import compute_hv


# =============================================================================
# DEFAULT PROBLEM SETS
# =============================================================================

# Population-size scaling: representative problems from the main benchmark suite
POP_SCALING_PROBLEMS = ["zdt4", "dtlz2", "wfg2"]

# Canonical dimensions used in the paper benchmark
_ZDT_N_VAR = {"zdt4": 10}
_DTLZ_N_VAR = {"dtlz2": 12}
_WFG_N_VAR = {"wfg2": 24}


def _canonical_dims(problem_name: str) -> tuple[int, int]:
    name = problem_name.lower()
    if name == "zdt4":
        return _ZDT_N_VAR[name], 2
    if name == "dtlz2":
        return _DTLZ_N_VAR[name], 3
    if name == "wfg2":
        return _WFG_N_VAR[name], 2
    raise ValueError(f"Unsupported problem for scaling: '{problem_name}'")


def _as_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _as_str_env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None else str(raw)


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _build_nsgaii_config(*, pop_size: int, n_var: int) -> NSGAIIConfig:
    return (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("pm", prob=1.0 / n_var, eta=20.0)
        .selection("tournament")
        .build()
    )


def _run_single(
    *,
    experiment: str,
    problem_name: str,
    n_var: int,
    n_obj: int,
    pop_size: int,
    n_evals: int,
    seed: int,
    engine: str,
    hv_enabled: bool,
    numba_warmup_evals: int,
) -> dict[str, Any]:
    selection = make_problem_selection(problem_name, n_var=n_var, n_obj=n_obj)
    problem = selection.instantiate()
    algo_cfg = _build_nsgaii_config(pop_size=pop_size, n_var=n_var)

    if engine == "numba" and numba_warmup_evals > 0:
        warmup_budget = min(int(numba_warmup_evals), int(n_evals))
        _ = optimize(
            problem,
            algorithm="nsgaii",
            algorithm_config=algo_cfg,
            termination=("max_evaluations", warmup_budget),
            seed=seed,
            engine=engine,
        )

    start = time.perf_counter()
    res = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=algo_cfg,
        termination=("max_evaluations", n_evals),
        seed=seed,
        engine=engine,
    )
    elapsed = time.perf_counter() - start

    hv = float("nan")
    if hv_enabled and res.F is not None:
        hv = float(compute_hv(res.F, problem_name))

    return {
        "experiment": experiment,
        "problem": problem_name,
        "n_obj": int(n_obj),
        "n_var": int(n_var),
        "pop_size": int(pop_size),
        "n_evals": int(n_evals),
        "seed": int(seed),
        "engine": str(engine),
        "timing_policy": "warm" if (engine == "numba" and numba_warmup_evals > 0) else "",
        "runtime_seconds": float(elapsed),
        "runtime_per_eval": float(elapsed) / float(n_evals),
        "hypervolume": hv,
    }


def _run_numba_jit_policy(
    *,
    problem_name: str,
    n_var: int,
    n_obj: int,
    pop_size: int,
    n_evals: int,
    seed: int,
    timing_policy: str,
    warmup_evals: int,
) -> dict[str, Any]:
    selection = make_problem_selection(problem_name, n_var=n_var, n_obj=n_obj)
    problem = selection.instantiate()
    algo_cfg = _build_nsgaii_config(pop_size=pop_size, n_var=n_var)

    if timing_policy not in {"cold", "warm"}:
        raise ValueError(f"Invalid timing_policy: '{timing_policy}'")

    if timing_policy == "warm":
        warmup_budget = min(int(warmup_evals), int(n_evals))
        _ = optimize(
            problem,
            algorithm="nsgaii",
            algorithm_config=algo_cfg,
            termination=("max_evaluations", warmup_budget),
            seed=seed,
            engine="numba",
        )

    start = time.perf_counter()
    res = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=algo_cfg,
        termination=("max_evaluations", n_evals),
        seed=seed,
        engine="numba",
    )
    elapsed = time.perf_counter() - start

    hv = float("nan")
    if res.F is not None:
        hv = float(compute_hv(res.F, problem_name))

    return {
        "experiment": "jit_policy",
        "problem": problem_name,
        "n_obj": int(n_obj),
        "n_var": int(n_var),
        "pop_size": int(pop_size),
        "n_evals": int(n_evals),
        "seed": int(seed),
        "engine": "numba",
        "timing_policy": timing_policy,
        "runtime_seconds": float(elapsed),
        "runtime_per_eval": float(elapsed) / float(n_evals),
        "hypervolume": hv,
    }


def main() -> None:
    n_evals = _as_int_env("VAMOS_N_EVALS", 50000)
    n_seeds = _as_int_env("VAMOS_N_SEEDS", 30)
    n_jobs = int(os.environ.get("VAMOS_N_JOBS", max(1, (os.cpu_count() or 2) - 1)))

    engines = [e.strip().lower() for e in _parse_str_list(_as_str_env("VAMOS_SCALING_ENGINES", "numpy,numba"))]
    output_csv = Path(_as_str_env("VAMOS_SCALING_OUTPUT_CSV", str(ROOT_DIR / "experiments" / "scaling_vectorization.csv")))

    pop_enabled = bool(_as_int_env("VAMOS_SCALING_POP_ENABLED", 1))
    obj_enabled = bool(_as_int_env("VAMOS_SCALING_OBJ_ENABLED", 1))
    jit_enabled = bool(_as_int_env("VAMOS_SCALING_JIT_ENABLED", 1))

    pop_sizes = _parse_int_list(_as_str_env("VAMOS_SCALING_POP_SIZES", "50,100,200,400,800"))
    obj_counts = _parse_int_list(_as_str_env("VAMOS_SCALING_OBJ_COUNTS", "2,3,5,8"))
    obj_pop_factor = _as_int_env("VAMOS_SCALING_OBJ_POP_FACTOR", 25)

    numba_warmup_evals = _as_int_env("VAMOS_SCALING_NUMBA_WARMUP_EVALS", 2000)
    jit_problems = _parse_str_list(_as_str_env("VAMOS_SCALING_JIT_PROBLEMS", ",".join(POP_SCALING_PROBLEMS)))
    jit_pop_size = _as_int_env("VAMOS_SCALING_JIT_POP_SIZE", 100)

    print("Scaling experiment configuration")
    print(f"- n_evals: {n_evals:,}")
    print(f"- n_seeds: {n_seeds}")
    print(f"- engines: {engines}")
    print(f"- workers: {n_jobs}")
    print(f"- output: {output_csv}")
    print(f"- numba warmup evals: {numba_warmup_evals}")
    print(f"- pop scaling enabled: {pop_enabled} (pop_sizes={pop_sizes})")
    print(f"- objective scaling enabled: {obj_enabled} (n_obj={obj_counts}, pop_factor={obj_pop_factor})")
    print(f"- jit policy enabled: {jit_enabled} (jit_problems={jit_problems}, jit_pop_size={jit_pop_size})")

    tasks: list[dict[str, Any]] = []

    if pop_enabled:
        for problem in POP_SCALING_PROBLEMS:
            n_var, n_obj = _canonical_dims(problem)
            for pop_size in pop_sizes:
                for seed in range(n_seeds):
                    for engine in engines:
                        tasks.append(
                            {
                                "experiment": "population",
                                "problem_name": problem,
                                "n_var": n_var,
                                "n_obj": n_obj,
                                "pop_size": pop_size,
                                "n_evals": n_evals,
                                "seed": seed,
                                "engine": engine,
                                "hv_enabled": True,
                                "numba_warmup_evals": numba_warmup_evals,
                            }
                        )

    if obj_enabled:
        for n_obj in obj_counts:
            n_var = int(n_obj) + 9  # DTLZ2 canonical: n = m + k - 1, with k=10
            pop_size = max(2, int(obj_pop_factor) * int(n_obj))
            for seed in range(n_seeds):
                for engine in engines:
                    tasks.append(
                        {
                            "experiment": "objectives",
                            "problem_name": "dtlz2",
                            "n_var": n_var,
                            "n_obj": int(n_obj),
                            "pop_size": pop_size,
                            "n_evals": n_evals,
                            "seed": seed,
                            "engine": engine,
                            "hv_enabled": False,
                            "numba_warmup_evals": numba_warmup_evals,
                        }
                    )

    print(f"Total runs: {len(tasks)}")

    rows = Parallel(n_jobs=n_jobs)(delayed(_run_single)(**task) for task in tasks)

    if jit_enabled:
        jit_tasks: list[dict[str, Any]] = []
        warmup_evals = max(0, int(numba_warmup_evals))
        for problem in jit_problems:
            n_var, n_obj = _canonical_dims(problem)
            for seed in range(n_seeds):
                for policy in ("cold", "warm"):
                    jit_tasks.append(
                        {
                            "problem_name": problem,
                            "n_var": n_var,
                            "n_obj": n_obj,
                            "pop_size": int(jit_pop_size),
                            "n_evals": int(n_evals),
                            "seed": int(seed),
                            "timing_policy": policy,
                            "warmup_evals": warmup_evals,
                        }
                    )

        print(f"JIT policy runs: {len(jit_tasks)} (engine=numba, max_tasks_per_child=1)")
        with ProcessPoolExecutor(max_workers=n_jobs, max_tasks_per_child=1) as ex:
            futures = [ex.submit(_run_numba_jit_policy, **task) for task in jit_tasks]
            for fut in as_completed(futures):
                rows.append(fut.result())

    df = pd.DataFrame(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} rows to {output_csv}")


if __name__ == "__main__":
    main()
