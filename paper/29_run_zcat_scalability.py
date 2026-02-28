"""
VAMOS Paper ZCAT Scalability Benchmark
=======================================
Runs NSGA-II on ZCAT1-20 with varying objective counts (2, 3, 4, 6) across
VAMOS (numba), pymoo, and jMetalPy.  Measures HV and IGD+ quality indicators.

All frameworks evaluate the same VAMOS ZCAT problem definitions to ensure
cross-framework alignment by construction.

Usage: python paper/29_run_zcat_scalability.py

Environment variables:
  VAMOS_N_EVALS              evaluations per run (default: 50000)
  VAMOS_N_SEEDS              independent seeds (default: 30)
  VAMOS_N_JOBS               parallel workers (default: cpu_count - 1)
  VAMOS_ZCAT_PROBLEMS        comma-separated problem IDs 1-20 (default: all)
  VAMOS_ZCAT_OBJECTIVES      comma-separated objective counts (default: 2,3,4,6)
  VAMOS_ZCAT_N_VAR           decision variables (default: 30)
  VAMOS_ZCAT_FRAMEWORKS      frameworks (default: vamos-numba,pymoo,jmetalpy)
  VAMOS_ZCAT_OUTPUT_CSV      output path (default: experiments/benchmark_zcat_scalability.csv)
  VAMOS_NUMBA_WARMUP_EVALS   Numba warmup evals (default: 2000)
  VAMOS_ZCAT_SAVE_EVERY      chunk size for partial saves (default: 50)
  VAMOS_ZCAT_SAVE_INTERVAL_MIN  time-based checkpoint interval in minutes (default: 30)
  VAMOS_ZCAT_RESUME          1/0 resume from existing CSV (default: 1)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from vamos import optimize
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.foundation.problem.registry import make_problem_selection

try:
    from .benchmark_utils import compute_hv, compute_igd_plus
except ImportError:
    from benchmark_utils import compute_hv, compute_igd_plus

try:
    from .progress_utils import ProgressBar, joblib_progress
except ImportError:  # pragma: no cover
    from progress_utils import ProgressBar, joblib_progress

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = ROOT_DIR / "experiments"


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


N_EVALS = _int_env("VAMOS_N_EVALS", 50_000)
N_SEEDS = _int_env("VAMOS_N_SEEDS", 30)
N_JOBS = _int_env("VAMOS_N_JOBS", max(1, (os.cpu_count() or 2) - 1))
NUMBA_WARMUP_EVALS = _int_env("VAMOS_NUMBA_WARMUP_EVALS", 2000)
SAVE_EVERY = _int_env("VAMOS_ZCAT_SAVE_EVERY", 50)
SAVE_INTERVAL_MIN = _int_env("VAMOS_ZCAT_SAVE_INTERVAL_MIN", 30)
RESUME = os.environ.get("VAMOS_ZCAT_RESUME", "1").strip().lower() not in {"0", "false", "no"}

# Problem configuration
N_VAR = _int_env("VAMOS_ZCAT_N_VAR", 30)
_problems_raw = os.environ.get("VAMOS_ZCAT_PROBLEMS", ",".join(str(i) for i in range(1, 21)))
PROBLEM_IDS = _parse_int_list(_problems_raw)
_obj_raw = os.environ.get("VAMOS_ZCAT_OBJECTIVES", "2,3,4,6")
OBJECTIVE_COUNTS = _parse_int_list(_obj_raw)

# Operator parameters (same as 01_run_paper_benchmark.py)
POP_SIZE = 100
CROSSOVER_PROB = 1.0
CROSSOVER_ETA = 20.0
MUTATION_ETA = 20.0

# Frameworks
_fw_raw = os.environ.get("VAMOS_ZCAT_FRAMEWORKS", "vamos-numba,pymoo,jmetalpy")
FRAMEWORKS = _parse_str_list(_fw_raw)

OUTPUT_CSV = Path(
    os.environ.get("VAMOS_ZCAT_OUTPUT_CSV", str(DATA_DIR / "benchmark_zcat_scalability.csv"))
)


# =============================================================================
# HELPERS
# =============================================================================


def ref_front_name(pid: int, n_obj: int) -> str:
    """Reference front name for benchmark_utils (maps to CSV filename)."""
    if n_obj == 2:
        return f"zcat{pid}"
    return f"zcat{pid}.{n_obj}d"


def _build_nsgaii(n_var: int) -> NSGAIIConfig:
    return (
        NSGAIIConfig.builder()
        .pop_size(POP_SIZE)
        .offspring_size(POP_SIZE)
        .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
        .mutation("polynomial", prob=1.0 / n_var, eta=MUTATION_ETA)
        .selection("tournament")
        .build()
    )


def _make_vamos_zcat(pid: int, n_var: int, n_obj: int):
    """Instantiate a VAMOS ZCAT problem."""
    return make_problem_selection(f"zcat{pid}", n_var=n_var, n_obj=n_obj).instantiate()


# =============================================================================
# FRAMEWORK PROBLEM WRAPPERS (use VAMOS ZCAT evaluation as backend)
# =============================================================================


def _make_pymoo_zcat(pid: int, n_var: int, n_obj: int):
    """Create a pymoo Problem wrapping VAMOS ZCAT evaluation."""
    from pymoo.core.problem import Problem as PymooProblem

    vamos_prob = _make_vamos_zcat(pid, n_var, n_obj)
    xl = np.asarray(vamos_prob.xl, dtype=float)
    xu = np.asarray(vamos_prob.xu, dtype=float)

    class _ZCATWrapper(PymooProblem):
        def __init__(self):
            super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
            self._vamos = vamos_prob

        def _evaluate(self, x, out, *args, **kwargs):
            F = np.zeros((x.shape[0], n_obj), dtype=float)
            o = {"F": F}
            self._vamos.evaluate(x, o)
            out["F"] = o["F"]

    return _ZCATWrapper()


def _make_jmetal_zcat(pid: int, n_var: int, n_obj: int):
    """Create a jMetalPy FloatProblem wrapping VAMOS ZCAT evaluation."""
    from jmetal.core.problem import FloatProblem

    vamos_prob = _make_vamos_zcat(pid, n_var, n_obj)
    xl = np.asarray(vamos_prob.xl, dtype=float).tolist()
    xu = np.asarray(vamos_prob.xu, dtype=float).tolist()

    class _ZCATWrapper(FloatProblem):
        def __init__(self):
            super().__init__()
            self.lower_bound = xl
            self.upper_bound = xu
            self._vamos = vamos_prob

        def name(self) -> str:
            return f"ZCAT{pid}"

        def number_of_objectives(self) -> int:
            return n_obj

        def number_of_constraints(self) -> int:
            return 0

        def number_of_variables(self) -> int:
            return n_var

        def evaluate(self, solution):
            x = np.asarray(solution.variables, dtype=float).reshape(1, -1)
            out = {"F": np.zeros((1, n_obj), dtype=float)}
            self._vamos.evaluate(x, out)
            solution.objectives = out["F"][0].tolist()
            return solution

    return _ZCATWrapper()


# =============================================================================
# SINGLE BENCHMARK RUN
# =============================================================================


def run_single_benchmark(
    pid: int,
    n_obj: int,
    seed: int,
    framework: str,
) -> dict[str, Any] | None:
    """Run one (problem, n_obj, seed, framework) combination."""
    problem_label = f"zcat{pid}"
    ref_name = ref_front_name(pid, n_obj)
    n_var = N_VAR
    result_entry = None

    # --- VAMOS ---
    if framework.startswith("vamos-"):
        backend = framework.replace("vamos-", "")
        try:
            problem = _make_vamos_zcat(pid, n_var, n_obj)
            algo_cfg = _build_nsgaii(n_var)

            if backend == "numba" and NUMBA_WARMUP_EVALS > 0:
                warmup_budget = min(NUMBA_WARMUP_EVALS, N_EVALS)
                _ = optimize(
                    problem,
                    algorithm="nsgaii",
                    algorithm_config=algo_cfg,
                    termination=("max_evaluations", warmup_budget),
                    seed=seed,
                    engine=backend,
                )

            start = time.perf_counter()
            res = optimize(
                problem,
                algorithm="nsgaii",
                algorithm_config=algo_cfg,
                termination=("max_evaluations", N_EVALS),
                seed=seed,
                engine=backend,
            )
            elapsed = time.perf_counter() - start

            hv = compute_hv(res.F, ref_name) if res.F is not None else float("nan")
            igd = compute_igd_plus(res.F, ref_name) if res.F is not None else float("nan")
            n_sol = res.X.shape[0] if res.X is not None else 0

            result_entry = {
                "framework": f"VAMOS ({backend})",
                "problem": problem_label,
                "n_obj": n_obj,
                "n_var": n_var,
                "algorithm": "NSGA-II",
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": n_sol,
                "hypervolume": hv,
                "igd_plus": igd,
            }
            print(f"  {problem_label} m={n_obj} VAMOS({backend}) seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_label} m={n_obj} VAMOS({backend}) seed={seed} FAILED: {e}")

    # --- pymoo ---
    elif framework == "pymoo":
        try:
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.operators.crossover.sbx import SBX
            from pymoo.operators.mutation.pm import PM
            from pymoo.optimize import minimize
            from pymoo.termination import get_termination

            pymoo_problem = _make_pymoo_zcat(pid, n_var, n_obj)

            algorithm = NSGA2(
                pop_size=POP_SIZE,
                crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
                mutation=PM(prob=1.0, prob_var=1.0 / n_var, eta=MUTATION_ETA),
            )
            termination = get_termination("n_eval", N_EVALS)

            start = time.perf_counter()
            res = minimize(pymoo_problem, algorithm, termination, seed=seed, verbose=False)
            elapsed = time.perf_counter() - start

            hv = compute_hv(res.F, ref_name) if res.F is not None else float("nan")
            igd = compute_igd_plus(res.F, ref_name) if res.F is not None else float("nan")
            n_sol = res.X.shape[0] if res.X is not None else 0

            result_entry = {
                "framework": "pymoo",
                "problem": problem_label,
                "n_obj": n_obj,
                "n_var": n_var,
                "algorithm": "NSGA-II",
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": n_sol,
                "hypervolume": hv,
                "igd_plus": igd,
            }
            print(f"  {problem_label} m={n_obj} pymoo seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_label} m={n_obj} pymoo seed={seed} FAILED: {e}")

    # --- jMetalPy ---
    elif framework == "jmetalpy":
        try:
            import random

            from jmetal.algorithm.multiobjective import NSGAII
            from jmetal.operator.crossover import SBXCrossover
            from jmetal.operator.mutation import PolynomialMutation
            from jmetal.util.termination_criterion import StoppingByEvaluations

            random.seed(seed)
            np.random.seed(seed)
            try:
                from jmetal.util.random_generator import PRNG
                PRNG.seed(seed)
            except Exception:
                pass

            jmetal_prob = _make_jmetal_zcat(pid, n_var, n_obj)

            algorithm = NSGAII(
                problem=jmetal_prob,
                population_size=POP_SIZE,
                offspring_population_size=POP_SIZE,
                mutation=PolynomialMutation(
                    probability=1.0 / n_var, distribution_index=MUTATION_ETA
                ),
                crossover=SBXCrossover(
                    probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA
                ),
                termination_criterion=StoppingByEvaluations(max_evaluations=N_EVALS),
            )

            start = time.perf_counter()
            algorithm.run()
            elapsed = time.perf_counter() - start

            solutions = algorithm.result()
            F = np.array([s.objectives for s in solutions])
            hv = compute_hv(F, ref_name)
            igd = compute_igd_plus(F, ref_name)

            result_entry = {
                "framework": "jMetalPy",
                "problem": problem_label,
                "n_obj": n_obj,
                "n_var": n_var,
                "algorithm": "NSGA-II",
                "n_evals": N_EVALS,
                "seed": seed,
                "runtime_seconds": elapsed,
                "n_solutions": len(solutions),
                "hypervolume": hv,
                "igd_plus": igd,
            }
            print(f"  {problem_label} m={n_obj} jMetalPy seed={seed}: {elapsed:.2f}s")
        except Exception as e:
            print(f"  {problem_label} m={n_obj} jMetalPy seed={seed} FAILED: {e}")

    else:
        print(f"  Unknown framework: {framework}")

    return result_entry


# =============================================================================
# MAIN
# =============================================================================


def _save_partial(collected: list[dict[str, Any]]) -> None:
    """Persist current results to CSV (existing + new rows already merged in memory)."""
    rows = [r for r in collected if r is not None]
    if not rows:
        return
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)


def _framework_csv_name(fw: str) -> str:
    """Map framework key to the name stored in CSV."""
    if fw.startswith("vamos-"):
        return f"VAMOS ({fw.replace('vamos-', '')})"
    if fw == "jmetalpy":
        return "jMetalPy"
    return fw


def main() -> None:
    print("ZCAT Scalability Benchmark")
    print(f"  Problems:    ZCAT{PROBLEM_IDS}")
    print(f"  Objectives:  {OBJECTIVE_COUNTS}")
    print(f"  n_var:       {N_VAR}")
    print(f"  n_evals:     {N_EVALS:,}")
    print(f"  seeds:       {N_SEEDS}")
    print(f"  frameworks:  {FRAMEWORKS}")
    print(f"  workers:     {N_JOBS}")
    print(f"  save every:  {SAVE_EVERY} tasks or {SAVE_INTERVAL_MIN} min")
    print(f"  output:      {OUTPUT_CSV}")

    # Build task list
    tasks: list[dict[str, Any]] = []
    for pid in PROBLEM_IDS:
        for n_obj in OBJECTIVE_COUNTS:
            for seed in range(N_SEEDS):
                for fw in FRAMEWORKS:
                    tasks.append(
                        {"pid": pid, "n_obj": n_obj, "seed": seed, "framework": fw}
                    )

    # Resume: skip already completed runs
    done_keys: set[tuple[str, int, int, int]] = set()
    existing_rows: list[dict[str, Any]] = []
    if RESUME and OUTPUT_CSV.is_file():
        try:
            df_existing = pd.read_csv(OUTPUT_CSV)
            for _, row in df_existing.iterrows():
                done_keys.add((
                    str(row["framework"]),
                    int(row["n_obj"]),
                    int(row["seed"]),
                    int(str(row["problem"]).replace("zcat", "")),
                ))
            existing_rows = df_existing.to_dict("records")
            print(f"Resuming: {len(done_keys)} runs already completed")
        except Exception as e:
            print(f"Warning: could not load existing CSV for resume: {e}")

    pending = [
        t for t in tasks
        if (_framework_csv_name(t["framework"]), t["n_obj"], t["seed"], t["pid"])
        not in done_keys
    ]

    total = len(pending)
    print(f"Total runs: {total} ({len(tasks) - total} skipped via resume)")
    if total == 0:
        print("Nothing to do.")
        return

    # Execute in chunks with time-based checkpointing
    collected: list[dict[str, Any]] = list(existing_rows)
    last_save = time.perf_counter()
    save_interval = SAVE_INTERVAL_MIN * 60

    for i in range(0, len(pending), SAVE_EVERY):
        chunk = pending[i : i + SAVE_EVERY]
        with joblib_progress(total=len(chunk), desc=f"ZCAT scalability [{i}..{i+len(chunk)}/{total}]"):
            chunk_results = Parallel(n_jobs=N_JOBS, batch_size=1)(
                delayed(run_single_benchmark)(**t) for t in chunk
            )
        for r in chunk_results:
            if r is not None:
                collected.append(r)

        now = time.perf_counter()
        if now - last_save >= save_interval:
            _save_partial(collected)
            n_new = len(collected) - len(existing_rows)
            print(f"  [checkpoint] saved {len(collected)} rows ({n_new} new) at {(now - last_save)/60:.1f}min")
            last_save = now

    # Final save
    _save_partial(collected)
    n_new = len(collected) - len(existing_rows)
    print(f"\nWrote {len(collected)} rows ({n_new} new) to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
