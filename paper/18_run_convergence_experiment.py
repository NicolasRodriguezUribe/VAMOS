"""
Convergence Tracking Experiment for VAMOS Paper
================================================
Records normalized hypervolume at regular evaluation checkpoints during NSGA-II
optimization on representative problems, comparing VAMOS (Numba) vs pymoo.

Usage: python paper/18_run_convergence_experiment.py

Environment variables:
  - VAMOS_CONV_PROBLEMS: comma-separated problems (default: zdt1,dtlz2,wfg4)
  - VAMOS_CONV_N_EVALS: total evaluations (default: 50000)
  - VAMOS_CONV_N_SEEDS: number of seeds (default: 30)
  - VAMOS_CONV_CHECKPOINT_EVERY: checkpoint interval in evaluations (default: 1000)

Output: experiments/convergence_paper.csv
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
if JMETALPY_SRC.exists():
    sys.path.insert(0, str(JMETALPY_SRC))

try:
    from .benchmark_utils import compute_hv
except ImportError:
    from benchmark_utils import compute_hv

# =============================================================================
# Configuration
# =============================================================================

PROBLEMS = os.environ.get("VAMOS_CONV_PROBLEMS", "zdt1,dtlz2,wfg4").strip().split(",")
N_EVALS = int(os.environ.get("VAMOS_CONV_N_EVALS", "50000"))
N_SEEDS = int(os.environ.get("VAMOS_CONV_N_SEEDS", "30"))
CHECKPOINT_EVERY = int(os.environ.get("VAMOS_CONV_CHECKPOINT_EVERY", "1000"))
POP_SIZE = 100

OUTPUT_CSV = ROOT_DIR / "experiments" / "convergence_paper.csv"

# Problem dimensions (matching 01_run_paper_benchmark.py)
ZDT_N_VAR = {"zdt1": 30, "zdt2": 30, "zdt3": 30, "zdt4": 10, "zdt6": 10}
DTLZ_N_VAR = {"dtlz1": 7, "dtlz2": 12, "dtlz3": 12, "dtlz4": 12, "dtlz5": 12, "dtlz6": 12, "dtlz7": 22}
WFG_N_VAR = 24
ZDT_N_OBJ = 2
DTLZ_N_OBJ = 3
WFG_N_OBJ = 2

# Operator parameters (matching benchmark)
CROSSOVER_PROB = 1.0
CROSSOVER_ETA = 20.0
MUTATION_ETA = 20.0


def get_problem_dims(problem_name: str) -> tuple[int, int]:
    if problem_name in ZDT_N_VAR:
        return ZDT_N_VAR[problem_name], ZDT_N_OBJ
    if problem_name in DTLZ_N_VAR:
        return DTLZ_N_VAR[problem_name], DTLZ_N_OBJ
    if problem_name.startswith("wfg"):
        return WFG_N_VAR, WFG_N_OBJ
    raise ValueError(f"Unknown problem: {problem_name}")


# =============================================================================
# VAMOS convergence callback
# =============================================================================

class VAMOSConvergenceCallback:
    """A live_viz-compatible callback that records HV at checkpoints."""

    def __init__(self, problem_name: str, checkpoint_every: int):
        self.problem_name = problem_name
        self.checkpoint_every = checkpoint_every
        self.trace: list[tuple[int, float]] = []
        self._last_checkpoint = 0

    def on_start(self, ctx=None):
        pass

    def on_generation(self, generation, F=None, X=None, stats=None):
        if stats is None or F is None:
            return
        evals = stats.get("evals", stats.get("n_evals", stats.get("evaluations", 0)))
        if evals is None:
            evals = (generation + 1) * POP_SIZE
        evals = int(evals)

        # Record at checkpoint boundaries
        while self._last_checkpoint + self.checkpoint_every <= evals:
            self._last_checkpoint += self.checkpoint_every
            hv = compute_hv(F, self.problem_name)
            self.trace.append((self._last_checkpoint, hv))

    def on_end(self, F=None, stats=None, **kwargs):
        pass

    def should_stop(self) -> bool:
        return False


# =============================================================================
# pymoo convergence callback
# =============================================================================

class PymooConvergenceCallback:
    """pymoo Callback that records HV at evaluation checkpoints."""

    def __init__(self, problem_name: str, checkpoint_every: int):
        self.problem_name = problem_name
        self.checkpoint_every = checkpoint_every
        self.trace: list[tuple[int, float]] = []
        self._last_checkpoint = 0

    def __call__(self, algorithm):
        """Called by pymoo after each generation."""
        evals = algorithm.evaluator.n_eval if hasattr(algorithm.evaluator, "n_eval") else 0
        F = algorithm.pop.get("F") if algorithm.pop is not None else None
        if F is None:
            return

        while self._last_checkpoint + self.checkpoint_every <= evals:
            self._last_checkpoint += self.checkpoint_every
            hv = compute_hv(F, self.problem_name)
            self.trace.append((self._last_checkpoint, hv))


# =============================================================================
# jMetalPy convergence via subclass
# =============================================================================

class JMetalPyConvergenceNSGAII:
    """Wraps jMetalPy NSGAII to record HV at evaluation checkpoints."""

    def __init__(self, problem_name: str, checkpoint_every: int):
        self.problem_name = problem_name
        self.checkpoint_every = checkpoint_every
        self.trace: list[tuple[int, float]] = []
        self._last_checkpoint = 0

    def run(self, problem_name: str, seed: int) -> list[tuple[int, float]]:
        import random
        from jmetal.algorithm.multiobjective import NSGAII
        from jmetal.operator.crossover import SBXCrossover
        from jmetal.operator.mutation import PolynomialMutation
        from jmetal.util.termination_criterion import StoppingByEvaluations
        from jmetal.problem import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
        from jmetal.problem import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
        from jmetal.problem import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9

        random.seed(seed)
        np.random.seed(seed)
        try:
            from jmetal.util.random_generator import PRNG
            PRNG.seed(seed)
        except Exception:
            pass

        n_var, n_obj = get_problem_dims(problem_name)
        wfg_k = 4 if n_obj == 2 else 2 * (n_obj - 1)
        wfg_l = WFG_N_VAR - wfg_k

        problem_map = {
            "zdt1": ZDT1(number_of_variables=ZDT_N_VAR["zdt1"]),
            "zdt2": ZDT2(number_of_variables=ZDT_N_VAR["zdt2"]),
            "zdt3": ZDT3(number_of_variables=ZDT_N_VAR["zdt3"]),
            "zdt4": ZDT4(number_of_variables=ZDT_N_VAR["zdt4"]),
            "zdt6": ZDT6(number_of_variables=ZDT_N_VAR["zdt6"]),
            "dtlz1": DTLZ1(number_of_variables=DTLZ_N_VAR["dtlz1"], number_of_objectives=DTLZ_N_OBJ),
            "dtlz2": DTLZ2(number_of_variables=DTLZ_N_VAR["dtlz2"], number_of_objectives=DTLZ_N_OBJ),
            "dtlz3": DTLZ3(number_of_variables=DTLZ_N_VAR["dtlz3"], number_of_objectives=DTLZ_N_OBJ),
            "dtlz4": DTLZ4(number_of_variables=DTLZ_N_VAR["dtlz4"], number_of_objectives=DTLZ_N_OBJ),
            "dtlz5": DTLZ5(number_of_variables=DTLZ_N_VAR["dtlz5"], number_of_objectives=DTLZ_N_OBJ),
            "dtlz6": DTLZ6(number_of_variables=DTLZ_N_VAR["dtlz6"], number_of_objectives=DTLZ_N_OBJ),
            "dtlz7": DTLZ7(number_of_variables=DTLZ_N_VAR["dtlz7"], number_of_objectives=DTLZ_N_OBJ),
            "wfg1": WFG1(number_of_variables=WFG_N_VAR, number_of_objectives=n_obj, k=wfg_k, l=wfg_l),
            "wfg2": WFG2(number_of_variables=WFG_N_VAR, number_of_objectives=n_obj, k=wfg_k, l=wfg_l),
            "wfg3": WFG3(number_of_variables=WFG_N_VAR, number_of_objectives=n_obj, k=wfg_k, l=wfg_l),
            "wfg4": WFG4(number_of_variables=WFG_N_VAR, number_of_objectives=n_obj, k=wfg_k, l=wfg_l),
            "wfg5": WFG5(number_of_variables=WFG_N_VAR, number_of_objectives=n_obj, k=wfg_k, l=wfg_l),
            "wfg6": WFG6(number_of_variables=WFG_N_VAR, number_of_objectives=n_obj, k=wfg_k, l=wfg_l),
            "wfg7": WFG7(number_of_variables=WFG_N_VAR, number_of_objectives=n_obj, k=wfg_k, l=wfg_l),
            "wfg8": WFG8(number_of_variables=WFG_N_VAR, number_of_objectives=n_obj, k=wfg_k, l=wfg_l),
            "wfg9": WFG9(number_of_variables=WFG_N_VAR, number_of_objectives=n_obj, k=wfg_k, l=wfg_l),
        }

        jmetal_problem = problem_map[problem_name]

        tracker = self
        _checkpoint_every = self.checkpoint_every
        _problem_name = self.problem_name

        class _TrackedNSGAII(NSGAII):
            def update_progress(self_algo):
                super().update_progress()
                evals = self_algo.evaluations
                while tracker._last_checkpoint + _checkpoint_every <= evals:
                    tracker._last_checkpoint += _checkpoint_every
                    F = np.array([s.objectives for s in self_algo.solutions])
                    hv = compute_hv(F, _problem_name)
                    tracker.trace.append((tracker._last_checkpoint, hv))

        algorithm = _TrackedNSGAII(
            problem=jmetal_problem,
            population_size=POP_SIZE,
            offspring_population_size=POP_SIZE,
            mutation=PolynomialMutation(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
            crossover=SBXCrossover(probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA),
            termination_criterion=StoppingByEvaluations(max_evaluations=N_EVALS),
        )

        algorithm.run()

        # Ensure final checkpoint
        if self.trace and self.trace[-1][0] < N_EVALS:
            F = np.array([s.objectives for s in algorithm.result()])
            hv = compute_hv(F, problem_name)
            self.trace.append((N_EVALS, hv))

        return self.trace


# =============================================================================
# Run functions
# =============================================================================

def run_vamos(problem_name: str, seed: int) -> list[tuple[int, float]]:
    """Run VAMOS NSGA-II and return convergence trace [(n_evals, hv), ...]."""
    from vamos.foundation.problem.registry import make_problem_selection
    from vamos import optimize
    from vamos.engine.algorithm.config import NSGAIIConfig

    n_var, n_obj = get_problem_dims(problem_name)
    problem = make_problem_selection(problem_name, n_var=n_var, n_obj=n_obj).instantiate()

    algo_config = (
        NSGAIIConfig.builder()
        .pop_size(POP_SIZE)
        .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
        .mutation("polynomial", prob=1.0 / n_var, eta=MUTATION_ETA)
        .selection("tournament")
        .build()
    )

    callback = VAMOSConvergenceCallback(problem_name, CHECKPOINT_EVERY)

    # Numba warmup (JIT compilation)
    _ = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=algo_config,
        termination=("max_evaluations", min(2000, N_EVALS)),
        seed=seed,
        engine="numba",
    )

    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=algo_config,
        termination=("max_evaluations", N_EVALS),
        seed=seed,
        engine="numba",
        live_viz=callback,
    )

    # Ensure we have the final checkpoint
    if callback.trace and callback.trace[-1][0] < N_EVALS:
        hv = compute_hv(result.F, problem_name)
        callback.trace.append((N_EVALS, hv))

    return callback.trace


def run_pymoo(problem_name: str, seed: int) -> list[tuple[int, float]]:
    """Run pymoo NSGA-II and return convergence trace [(n_evals, hv), ...]."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.problems import get_problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM

    n_var, n_obj = get_problem_dims(problem_name)
    if problem_name.startswith("zdt"):
        pymoo_problem = get_problem(problem_name, n_var=n_var)
    else:
        pymoo_problem = get_problem(problem_name, n_var=n_var, n_obj=n_obj)

    callback = PymooConvergenceCallback(problem_name, CHECKPOINT_EVERY)

    algorithm = NSGA2(
        pop_size=POP_SIZE,
        crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
        mutation=PM(prob=1.0 / n_var, eta=MUTATION_ETA),
    )

    termination = get_termination("n_eval", N_EVALS)

    res = minimize(
        pymoo_problem,
        algorithm,
        termination,
        seed=seed,
        verbose=False,
        callback=callback,
    )

    # Ensure final checkpoint
    if callback.trace and callback.trace[-1][0] < N_EVALS:
        hv = compute_hv(res.F, problem_name)
        callback.trace.append((N_EVALS, hv))

    return callback.trace


def run_jmetalpy(problem_name: str, seed: int) -> list[tuple[int, float]]:
    """Run jMetalPy NSGA-II and return convergence trace [(n_evals, hv), ...]."""
    tracker = JMetalPyConvergenceNSGAII(problem_name, CHECKPOINT_EVERY)
    return tracker.run(problem_name, seed)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("Convergence Tracking Experiment")
    print("=" * 60)
    print(f"Problems: {PROBLEMS}")
    print(f"Evaluations: {N_EVALS}")
    print(f"Seeds: {N_SEEDS}")
    print(f"Checkpoint every: {CHECKPOINT_EVERY} evaluations")
    print(f"Output: {OUTPUT_CSV}")
    print()

    all_results: list[dict] = []
    seeds = list(range(N_SEEDS))

    for problem_name in PROBLEMS:
        print(f"\n--- {problem_name} ---")
        n_var, n_obj = get_problem_dims(problem_name)
        print(f"  n_var={n_var}, n_obj={n_obj}")

        for seed in seeds:
            # VAMOS (Numba)
            try:
                trace = run_vamos(problem_name, seed)
                for n_evals, hv in trace:
                    all_results.append({
                        "framework": "VAMOS (numba)",
                        "problem": problem_name,
                        "seed": seed,
                        "n_evals": n_evals,
                        "hypervolume": hv,
                    })
                print(f"  VAMOS seed={seed}: {len(trace)} checkpoints, final HV={trace[-1][1]:.4f}" if trace else f"  VAMOS seed={seed}: no trace")
            except Exception as e:
                print(f"  VAMOS seed={seed} FAILED: {e}")

            # pymoo
            try:
                trace = run_pymoo(problem_name, seed)
                for n_evals, hv in trace:
                    all_results.append({
                        "framework": "pymoo",
                        "problem": problem_name,
                        "seed": seed,
                        "n_evals": n_evals,
                        "hypervolume": hv,
                    })
                print(f"  pymoo seed={seed}: {len(trace)} checkpoints, final HV={trace[-1][1]:.4f}" if trace else f"  pymoo seed={seed}: no trace")
            except Exception as e:
                print(f"  pymoo seed={seed} FAILED: {e}")

            # jMetalPy
            try:
                trace = run_jmetalpy(problem_name, seed)
                for n_evals, hv in trace:
                    all_results.append({
                        "framework": "jMetalPy",
                        "problem": problem_name,
                        "seed": seed,
                        "n_evals": n_evals,
                        "hypervolume": hv,
                    })
                print(f"  jMetalPy seed={seed}: {len(trace)} checkpoints, final HV={trace[-1][1]:.4f}" if trace else f"  jMetalPy seed={seed}: no trace")
            except Exception as e:
                print(f"  jMetalPy seed={seed} FAILED: {e}")

        # Save partial results after each problem
        df = pd.DataFrame(all_results)
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"  Saved {len(df)} rows to {OUTPUT_CSV}")

    print(f"\nDone! Total rows: {len(all_results)}")


if __name__ == "__main__":
    main()
