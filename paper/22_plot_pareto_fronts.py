"""
Pareto Front Comparison Plot
=============================
Generates a figure overlaying approximate Pareto fronts from all frameworks
on representative problems, alongside the true reference front.

Usage: python paper/22_plot_pareto_fronts.py

Environment variables:
  - VAMOS_PARETO_SEED: seed to use for representative fronts (default: 0)
  - VAMOS_N_EVALS: evaluations per run (default: 50000)

Output: paper/manuscript/figures/pareto_fronts.png
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

DESKTOP_DIR = ROOT_DIR.parent
JMETALPY_SRC = DESKTOP_DIR / "jMetalPy" / "src"
PLATYPUS_SRC = DESKTOP_DIR / "Platypus"
for extra_path in (JMETALPY_SRC, PLATYPUS_SRC):
    if extra_path.exists():
        sys.path.insert(0, str(extra_path))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vamos.foundation.metrics.pareto import pareto_filter

OUTPUT_FIG = Path(__file__).parent / "manuscript" / "figures" / "pareto_fronts.png"

SEED = int(os.environ.get("VAMOS_PARETO_SEED", "0"))
N_EVALS = int(os.environ.get("VAMOS_N_EVALS", "50000"))
POP_SIZE = 100
CROSSOVER_PROB = 1.0
CROSSOVER_ETA = 20.0
MUTATION_ETA = 20.0

# Reference front directory
REFERENCE_FRONTS_DIR = ROOT_DIR / "src" / "vamos" / "foundation" / "data" / "reference_fronts"

# Problems to plot: ZDT1 (2D scatter) and DTLZ2 (3D)
PROBLEMS_2D = [("zdt1", 30, 2)]
PROBLEMS_3D = [("dtlz2", 12, 3)]

FRAMEWORK_STYLES = {
    "VAMOS": {"color": "#1f77b4", "marker": "o", "ms": 3, "alpha": 0.7},
    "pymoo": {"color": "#ff7f0e", "marker": "s", "ms": 3, "alpha": 0.7},
    "jMetalPy": {"color": "#2ca02c", "marker": "^", "ms": 3, "alpha": 0.7},
    "DEAP": {"color": "#d62728", "marker": "D", "ms": 3, "alpha": 0.7},
    "Platypus": {"color": "#9467bd", "marker": "v", "ms": 3, "alpha": 0.7},
}


def load_reference_front(problem_name: str) -> np.ndarray:
    path = REFERENCE_FRONTS_DIR / f"{problem_name}.csv"
    if not path.exists():
        path = REFERENCE_FRONTS_DIR / f"{problem_name.upper()}.csv"
    return np.loadtxt(path, delimiter=",")


def get_vamos_front(problem_name: str, n_var: int, n_obj: int) -> np.ndarray | None:
    try:
        from vamos.foundation.problem.registry import make_problem_selection
        from vamos import optimize
        from vamos.engine.algorithm.config import NSGAIIConfig

        problem = make_problem_selection(problem_name, n_var=n_var, n_obj=n_obj).instantiate()
        cfg = (
            NSGAIIConfig.builder()
            .pop_size(POP_SIZE)
            .crossover("sbx", prob=CROSSOVER_PROB, eta=CROSSOVER_ETA)
            .mutation("pm", prob=1.0 / n_var, eta=MUTATION_ETA)
            .selection("tournament")
            .build()
        )
        result = optimize(problem, algorithm="nsgaii", algorithm_config=cfg,
                          termination=("max_evaluations", N_EVALS),
                          seed=SEED, engine="numba")
        return pareto_filter(result.F)
    except Exception as e:
        print(f"  VAMOS failed: {e}")
        return None


def get_pymoo_front(problem_name: str, n_var: int, n_obj: int) -> np.ndarray | None:
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
        from pymoo.problems import get_problem
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.selection.rnd import RandomSelection

        pymoo_problem = get_problem(problem_name)
        algorithm = NSGA2(
            pop_size=POP_SIZE,
            crossover=SBX(prob=CROSSOVER_PROB, eta=CROSSOVER_ETA),
            mutation=PM(prob=1.0 / n_var, eta=MUTATION_ETA),
            selection=RandomSelection(),
        )
        res = minimize(pymoo_problem, algorithm, get_termination("n_eval", N_EVALS),
                       seed=SEED, verbose=False)
        return pareto_filter(res.F)
    except Exception as e:
        print(f"  pymoo failed: {e}")
        return None


def get_jmetalpy_front(problem_name: str, n_var: int, n_obj: int) -> np.ndarray | None:
    try:
        from jmetal.algorithm.multiobjective import NSGAII
        from jmetal.operator.crossover import SBXCrossover
        from jmetal.operator.mutation import PolynomialMutation
        from jmetal.util.termination_criterion import StoppingByEvaluations
        from jmetal.problem.multiobjective.zdt import ZDT1
        from jmetal.problem.multiobjective.dtlz import DTLZ2
        import random

        jmetal_problems = {
            "zdt1": ZDT1(number_of_variables=n_var),
            "dtlz2": DTLZ2(number_of_variables=n_var, number_of_objectives=n_obj),
        }
        jmetal_problem = jmetal_problems.get(problem_name)
        if jmetal_problem is None:
            return None

        random.seed(SEED)
        algorithm = NSGAII(
            problem=jmetal_problem,
            population_size=POP_SIZE,
            offspring_population_size=POP_SIZE,
            crossover=SBXCrossover(probability=CROSSOVER_PROB, distribution_index=CROSSOVER_ETA),
            mutation=PolynomialMutation(probability=1.0 / n_var, distribution_index=MUTATION_ETA),
            termination_criterion=StoppingByEvaluations(max_evaluations=N_EVALS),
        )
        algorithm.run()
        solutions = algorithm.result()
        F = np.array([s.objectives for s in solutions])
        return pareto_filter(F)
    except Exception as e:
        print(f"  jMetalPy failed: {e}")
        return None


def plot_2d(ax, problem_name: str, n_var: int, n_obj: int):
    """Plot 2D Pareto front comparison."""
    # Reference front
    ref = load_reference_front(problem_name)
    ax.plot(ref[:, 0], ref[:, 1], 'k-', linewidth=0.8, alpha=0.4, label="Reference", zorder=0)

    # Framework fronts
    getters = {
        "VAMOS": get_vamos_front,
        "pymoo": get_pymoo_front,
        "jMetalPy": get_jmetalpy_front,
    }
    for fw_name, getter in getters.items():
        print(f"  Running {fw_name} on {problem_name}...")
        F = getter(problem_name, n_var, n_obj)
        if F is not None:
            style = FRAMEWORK_STYLES[fw_name]
            ax.scatter(F[:, 0], F[:, 1], s=style["ms"]**2, c=style["color"],
                       marker=style["marker"], alpha=style["alpha"], label=fw_name,
                       edgecolors="none", zorder=2)

    ax.set_xlabel("$f_1$", fontsize=10)
    ax.set_ylabel("$f_2$", fontsize=10)
    ax.set_title(f"{problem_name.upper()} (2-obj)", fontsize=11)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linewidth=0.5)


def plot_3d(ax, problem_name: str, n_var: int, n_obj: int):
    """Plot 3D Pareto front comparison with projections."""
    # Reference front
    ref = load_reference_front(problem_name)
    ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], s=1, c='gray', alpha=0.2, label="Reference")

    getters = {
        "VAMOS": get_vamos_front,
        "pymoo": get_pymoo_front,
        "jMetalPy": get_jmetalpy_front,
    }
    for fw_name, getter in getters.items():
        print(f"  Running {fw_name} on {problem_name}...")
        F = getter(problem_name, n_var, n_obj)
        if F is not None:
            style = FRAMEWORK_STYLES[fw_name]
            ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=style["ms"]**2, c=style["color"],
                       marker=style["marker"], alpha=style["alpha"], label=fw_name)

    ax.set_xlabel("$f_1$", fontsize=9, labelpad=2)
    ax.set_ylabel("$f_2$", fontsize=9, labelpad=2)
    ax.set_zlabel("$f_3$", fontsize=9, labelpad=2)
    ax.set_title(f"{problem_name.upper()} (3-obj)", fontsize=11)
    ax.tick_params(labelsize=7)


def main() -> None:
    print("Generating Pareto front comparison figure...")

    fig = plt.figure(figsize=(10, 4))

    # Left panel: ZDT1 (2D)
    ax1 = fig.add_subplot(1, 2, 1)
    for name, n_var, n_obj in PROBLEMS_2D:
        plot_2d(ax1, name, n_var, n_obj)

    # Right panel: DTLZ2 (3D)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for name, n_var, n_obj in PROBLEMS_3D:
        plot_3d(ax2, name, n_var, n_obj)

    # Shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
