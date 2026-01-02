from __future__ import annotations

import logging
import time

import numpy as np

from vamos.foundation.problem.registry import ProblemSelection


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _run_pygmo_nsga2(
    selection: ProblemSelection,
    *,
    use_native_problem: bool,
    config,
    make_metrics,
    print_banner,
    print_results,
):
    if selection.spec.key != "zdt1":
        raise ValueError("PyGMO baseline currently supports only ZDT1.")
    problem = selection.instantiate()
    print_banner(problem, selection, "PyGMO NSGA-II", "pygmo")
    try:
        import pygmo as pg
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pygmo is not installed. Install it with 'pip install pygmo' to use this baseline.") from exc

    generations = config.max_evaluations // config.population_size - 1
    uda = pg.nsga2(
        gen=generations,
        seed=config.seed,
        cr=0.9,
        eta_c=20.0,
        m=1.0 / selection.n_var,
        eta_m=20.0,
    )
    algo = pg.algorithm(uda)
    if use_native_problem:
        base_problem = pg.zdt(prob_id=1, dim=selection.n_var)
    else:

        class _VamosPyGMOProblem:
            def __init__(self, base_problem):
                self._base_problem = base_problem
                self._lower = [base_problem.xl] * base_problem.n_var
                self._upper = [base_problem.xu] * base_problem.n_var

            def fitness(self, x):
                X = np.asarray(x, dtype=float)[np.newaxis, :]
                F = np.empty((1, self._base_problem.n_obj))
                self._base_problem.evaluate(X, {"F": F})
                return F[0].tolist()

            def get_bounds(self):
                return (self._lower, self._upper)

            def get_nobj(self):
                return self._base_problem.n_obj

            def get_name(self):
                return "VAMOS-ZDT1"

        base_problem = _VamosPyGMOProblem(problem)

    pg_problem = pg.problem(base_problem)
    pop = pg.population(pg_problem, size=config.population_size, seed=config.seed)
    start = time.perf_counter()
    pop = algo.evolve(pop)
    end = time.perf_counter()
    total_eval = config.population_size * (generations + 1)
    total_time_ms = (end - start) * 1000.0
    F = np.asarray(pop.get_f(), dtype=float)
    metrics = make_metrics("pygmo_nsga2", "pygmo", total_time_ms, total_eval, F)
    print_results(metrics)
    _logger().info("%s", "=" * 80)
    return metrics
