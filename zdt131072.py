"""
ZDT1 runner with the AutoNSGA-II (external archive) settings and 131,072 evals.
Spawns one job per CPU core using the MooCore backend.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

# Allow running directly even if the package is not installed (e.g., via IDE play button)
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vamos.problem.registry import make_problem_selection
from vamos.runner import ExperimentConfig, run_single

# AutoNSGA-II (second column) settings mapped to VAMOS
POP_SIZE = 56
OFFSPRING_SIZE = 14
EXTERNAL_ARCHIVE_SIZE = 56
CROSSOVER = {"method": "blx_alpha", "prob": 0.88, "alpha": 0.94, "repair": "clip"}  # "bounds" -> clip
MUTATION = {"method": "non_uniform", "prob": "0.45/n", "perturbation": 0.3}
REPAIR_DEFAULT = "round"
SELECTION_PRESSURE = 9
MAX_EVALUATIONS = 19359356
ENGINE = "moocore"
# Set these to override CLI defaults directly in the file (hardcoded run params)
HARDCODE_N_VAR: int | None = 2048#131072  # e.g., 100 to force 100 variables
HARDCODE_REPAIR: str | None = None  # e.g., "clip", "round", "reflect", "random", or "none"
HARDCODE_SEEDS: list[int] | None = [1]  # e.g., [1, 2, 3] to control parallel runs


def _run_seed(seed: int, repair: str | None, n_var: int | None) -> dict[str, Any]:
    # Let MooCore/OpenMP use all cores unless the user overrides.
    os.environ.setdefault("OMP_NUM_THREADS", str(mp.cpu_count()))
    selection = make_problem_selection("zdt1", n_var=n_var)
    config = ExperimentConfig(
        population_size=POP_SIZE,
        offspring_population_size=OFFSPRING_SIZE,
        max_evaluations=MAX_EVALUATIONS,
        seed=seed,
    )
    repair_cfg = repair if repair != "none" else None
    return run_single(
        ENGINE,
        "nsgaii",
        selection,
        config,
        external_archive_size=EXTERNAL_ARCHIVE_SIZE,
        selection_pressure=SELECTION_PRESSURE,
        nsgaii_variation={
            "crossover": CROSSOVER,
            "mutation": MUTATION,
            "repair": repair_cfg,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ZDT1 with AutoNSGA-II settings at 131,072 evals.")
    parser.add_argument(
        "--repair",
        choices=("round", "clip", "reflect", "random", "none"),
        default=REPAIR_DEFAULT,
        help="NSGA-II repair operator (default: round; use 'none' to disable repair).",
    )
    parser.add_argument(
        "--n-var",
        type=int,
        default=None,
        help="Override the number of decision variables for ZDT1 (default: problem default).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    repair = HARDCODE_REPAIR if HARDCODE_REPAIR is not None else args.repair
    n_var = HARDCODE_N_VAR if HARDCODE_N_VAR is not None else args.n_var
    seeds = HARDCODE_SEEDS if HARDCODE_SEEDS is not None else list(range(1, mp.cpu_count() + 1))
    print(f"Running {len(seeds)} parallel jobs on {mp.cpu_count()} cores...")
    with mp.Pool() as pool:
        metrics = pool.starmap(_run_seed, [(seed, repair, n_var) for seed in seeds])
    print("Done. Outputs under results/ZDT1/nsgaii/moocore/seed_*")
    return metrics


if __name__ == "__main__":
    mp.freeze_support()
    main()
