"""
Diagnostic helpers for the internal NSGA-II implementation.

This module instruments the evolutionary loop to capture statistics per
generation and compare the internal behaviour against external references
such as PyMOO.  Run it directly to dump the collected data under
``study/diagnostics`` for further inspection (e.g., plotting in notebooks).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem
from vamos.foundation.core.experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)
_DEFAULT_CONFIG = ExperimentConfig()
POPULATION_SIZE = _DEFAULT_CONFIG.population_size
MAX_EVALUATIONS = _DEFAULT_CONFIG.max_evaluations
SEED = _DEFAULT_CONFIG.seed


DIAG_ROOT = Path(__file__).resolve().parent / "diagnostics"


def _ensure_diag_dir() -> Path:
    DIAG_ROOT.mkdir(parents=True, exist_ok=True)
    return DIAG_ROOT


def _build_internal_algorithm(engine: str = "numpy") -> Tuple[NSGAII, Dict]:
    cfg = (
        NSGAIIConfig()
        .pop_size(POPULATION_SIZE)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .survival("nsga2")
        .engine(engine)
    ).fixed()
    return NSGAII(cfg.to_dict(), kernel=NumPyKernel()), cfg.to_dict()


def _prepare_params(cfg_dict: dict, n_var: int) -> Tuple[dict, dict, int]:
    cross_method, cross_params = cfg_dict["crossover"]
    assert cross_method == "sbx", "Only SBX crossover is supported in diagnostics."
    cross_params = dict(cross_params)

    mut_method, mut_params = cfg_dict["mutation"]
    assert mut_method == "pm", "Only polynomial mutation is supported in diagnostics."
    mut_params = dict(mut_params)
    if mut_params.get("prob") == "1/n":
        mut_params["prob"] = 1.0 / n_var

    sel_method, sel_params = cfg_dict["selection"]
    assert sel_method == "tournament", "Diagnostics expect tournament selection."
    pressure = int(sel_params.get("pressure", 2))
    return cross_params, mut_params, pressure


def _initialize_population(rng: np.random.Generator, problem, pop_size: int) -> Tuple[np.ndarray, np.ndarray]:
    X = rng.uniform(problem.xl, problem.xu, size=(pop_size, problem.n_var))
    F = np.empty((pop_size, problem.n_obj))
    problem.evaluate(X, {"F": F})
    return X, F


def _log_stats(X: np.ndarray, F: np.ndarray) -> Dict:
    stats = {
        "f1_min": float(F[:, 0].min()),
        "f1_max": float(F[:, 0].max()),
        "f2_min": float(F[:, 1].min()),
        "f2_max": float(F[:, 1].max()),
        "f2_mean": float(F[:, 1].mean()),
    }
    if X.shape[1] > 1:
        tail = X[:, 1:]
        tail_means = tail.mean(axis=1)
        stats.update(
            {
                "tail_mean_min": float(tail_means.min()),
                "tail_mean_max": float(tail_means.max()),
                "tail_mean_avg": float(tail_means.mean()),
                "g_min": float(1.0 + 9.0 * tail_means.min()),
                "g_max": float(1.0 + 9.0 * tail_means.max()),
            }
        )
    return stats


def _evolution_loop(
    problem,
    algorithm: NSGAII,
    cfg_dict: dict,
    *,
    rng: np.random.Generator,
    max_eval: int,
    log_every: int = 1,
    X_init: Optional[np.ndarray] = None,
    F_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    pop_size = cfg_dict["pop_size"]
    if X_init is None or F_init is None:
        X, F = _initialize_population(rng, problem, pop_size)
        n_eval = pop_size
    else:
        X = np.asarray(X_init, dtype=float).copy()
        F = np.asarray(F_init, dtype=float).copy()
        n_eval = pop_size  # evaluate once per generation after this point

    cross_params, mut_params, pressure = _prepare_params(cfg_dict, problem.n_var)
    stats_log: List[Dict] = []
    generation = 0

    while n_eval < max_eval:
        if generation % log_every == 0:
            sample = {"generation": generation, "evaluations": int(n_eval)}
            sample.update(_log_stats(X, F))
            stats_log.append(sample)

        ranks, crowd = algorithm.kernel.nsga2_ranking(F)
        n_pairs = pop_size // 2
        parents_idx = algorithm.kernel.tournament_selection(ranks, crowd, pressure, rng, n_parents=2 * n_pairs)
        X_parents = X[parents_idx]
        X_off = algorithm.kernel.sbx_crossover(X_parents, cross_params, rng, problem.xl, problem.xu)
        algorithm.kernel.polynomial_mutation(X_off, mut_params, rng, problem.xl, problem.xu)

        F_off = np.empty((X_off.shape[0], problem.n_obj))
        problem.evaluate(X_off, {"F": F_off})
        n_eval += X_off.shape[0]
        X, F = algorithm.kernel.nsga2_survival(X, F, X_off, F_off, pop_size)
        generation += 1

    sample = {"generation": generation, "evaluations": int(n_eval)}
    sample.update(_log_stats(X, F))
    stats_log.append(sample)
    return X, F, stats_log


def run_progress_diagnostic(seed: int, max_eval: int, n_var: int) -> Dict:
    problem = ZDT1Problem(n_var=n_var)
    algorithm, cfg_dict = _build_internal_algorithm()
    rng = np.random.default_rng(seed)
    _, F, log = _evolution_loop(
        problem,
        algorithm,
        cfg_dict,
        rng=rng,
        max_eval=max_eval,
        log_every=1,
    )
    payload = {
        "seed": seed,
        "max_eval": max_eval,
        "n_var": n_var,
        "final_f2_min": float(F[:, 1].min()),
        "final_f2_max": float(F[:, 1].max()),
        "log": log,
    }
    return payload


def _pymoo_reference(seed: int, n_var: int, max_eval: int):
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2 as PymooNSGA2
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.optimize import minimize
        from pymoo.problems import get_problem
    except ImportError:
        return None

    pymoo_problem = get_problem("zdt1", n_var=n_var)
    crossover = SBX(prob=0.9, eta=20)
    mutation = PM(prob=1.0 / n_var, eta=20)
    algorithm = PymooNSGA2(
        pop_size=POPULATION_SIZE,
        sampling=FloatRandomSampling(),
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )
    res = minimize(
        pymoo_problem,
        algorithm,
        ("n_eval", max_eval),
        seed=seed,
        verbose=False,
    )
    X = np.asarray(res.X, dtype=float)
    F = np.asarray(res.F, dtype=float)
    return {"X": X, "F": F}


def compare_with_pymoo(seed: int, n_var: int, max_eval: int) -> Optional[Dict]:
    reference = _pymoo_reference(seed, n_var, max_eval)
    if reference is None:
        return None

    problem = ZDT1Problem(n_var=n_var)
    algorithm, cfg_dict = _build_internal_algorithm()
    rng = np.random.default_rng(seed)
    _, F_final, log = _evolution_loop(
        problem,
        algorithm,
        cfg_dict,
        rng=rng,
        max_eval=max_eval,
        X_init=reference["X"],
        F_init=reference["F"],
    )
    payload = {
        "seed": seed,
        "n_var": n_var,
        "max_eval": max_eval,
        "pymoo_initial_f2_min": float(reference["F"][:, 1].min()),
        "internal_after_pymoo_f2_min": float(F_final[:, 1].min()),
        "log": log,
    }
    return payload


def main():
    parser = argparse.ArgumentParser(description="Diagnostics for the internal NSGA-II implementation.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument(
        "--max-eval",
        type=int,
        default=MAX_EVALUATIONS,
        help="Total evaluation budget.",
    )
    parser.add_argument(
        "--n-var",
        type=int,
        default=30,
        help="Number of decision variables for ZDT1.",
    )
    parser.add_argument(
        "--skip-pymoo",
        action="store_true",
        help="Skip the PyMOO comparison even if pymoo is installed.",
    )
    args = parser.parse_args()

    output_dir = _ensure_diag_dir()
    progress = run_progress_diagnostic(args.seed, args.max_eval, args.n_var)
    progress_path = output_dir / f"nsga2_progress_seed{args.seed}_eval{args.max_eval}_nvar{args.n_var}.json"
    with open(progress_path, "w", encoding="utf-8") as fh:
        json.dump(progress, fh, indent=2)
    logger.info("Saved internal progress stats to %s", progress_path)

    if not args.skip_pymoo:
        compare = compare_with_pymoo(args.seed, args.n_var, args.max_eval)
        if compare is None:
            logger.info("pymoo is not installed; skipping external comparison.")
        else:
            compare_path = output_dir / f"nsga2_pymoo_compare_seed{args.seed}_eval{args.max_eval}_nvar{args.n_var}.json"
            with open(compare_path, "w", encoding="utf-8") as fh:
                json.dump(compare, fh, indent=2)
            logger.info("Saved PyMOO comparison stats to %s", compare_path)


if __name__ == "__main__":
    main()
