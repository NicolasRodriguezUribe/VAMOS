from __future__ import annotations

import logging
import time

import numpy as np

from vamos.foundation.problem.registry import ProblemSelection

logger = logging.getLogger(__name__)


def _run_pymoo_nsga2(
    selection: ProblemSelection,
    *,
    use_native_problem: bool,
    config,
    make_metrics,
    print_banner,
    print_results,
):
    if selection.spec.key != "zdt1":
        raise ValueError("PyMOO baseline currently supports only ZDT1.")
    problem = selection.instantiate()
    print_banner(problem, selection, "PyMOO NSGA-II", "pymoo")
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.optimize import minimize
        from pymoo.problems import get_problem
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pymoo is not installed. Install it with 'pip install pymoo' to use this baseline.") from exc

    if use_native_problem:
        pymoo_problem = get_problem("zdt1", n_var=selection.n_var)
    else:

        class _VamosPymooProblem(PymooProblem):
            def __init__(self, base_problem):
                super().__init__(
                    n_var=base_problem.n_var,
                    n_obj=base_problem.n_obj,
                    xl=base_problem.xl,
                    xu=base_problem.xu,
                )
                self._base_problem = base_problem

            def _evaluate(self, X, out, *args, **kwargs):
                F = np.empty((X.shape[0], self.n_obj))
                self._base_problem.evaluate(X, {"F": F})
                out["F"] = F

        pymoo_problem = _VamosPymooProblem(problem)

    mutation_prob = 1.0 / selection.n_var
    crossover = SBX(prob=0.9, eta=20)
    mutation = PM(prob=mutation_prob, eta=20)
    algorithm = NSGA2(
        pop_size=config.population_size,
        sampling=FloatRandomSampling(),
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )
    start = time.perf_counter()
    res = minimize(
        pymoo_problem,
        algorithm,
        ("n_eval", config.max_evaluations),
        seed=config.seed,
        verbose=False,
    )
    end = time.perf_counter()
    total_time_ms = (end - start) * 1000.0
    F = np.asarray(res.F, dtype=float)
    metrics = make_metrics("pymoo_nsga2", "pymoo", total_time_ms, config.max_evaluations, F)
    print_results(metrics)
    logger.info("%s", "=" * 80)
    return metrics


def _run_pymoo_perm_nsga2(
    selection: ProblemSelection,
    *,
    use_native_problem: bool,
    config,
    make_metrics,
    print_banner,
    print_results,
):
    problem = selection.instantiate()
    encoding = getattr(problem, "encoding", "continuous")
    if encoding != "permutation":
        raise ValueError("PyMOO permutation baseline requires a permutation-encoded problem.")
    print_banner(problem, selection, "PyMOO NSGA-II (perm)", "pymoo")
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.operators.sampling.rnd import PermutationRandomSampling
        from pymoo.operators.mutation.inversion import InversionMutation
        from pymoo.optimize import minimize
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pymoo is not installed. Install it with 'pip install pymoo' to use this baseline.") from exc

    class _VamosPymooPermutationProblem(PymooProblem):
        def __init__(self, base_problem):
            super().__init__(
                n_var=base_problem.n_var,
                n_obj=base_problem.n_obj,
                xl=0,
                xu=base_problem.n_var - 1,
                elementwise_evaluation=False,
            )
            self.base_problem = base_problem

        def _evaluate(self, X, out, *args, **kwargs):
            perms = np.asarray(X, dtype=int)
            F = np.empty((perms.shape[0], self.n_obj))
            self.base_problem.evaluate(perms, {"F": F})
            out["F"] = F

    pymoo_problem = _VamosPymooPermutationProblem(problem)
    mutation_prob = min(1.0, 2.0 / max(1, problem.n_var))

    def _make_pymoo_perm_crossover(probability: float):
        try:
            from pymoo.operators.crossover.pmx import PMX  # type: ignore

            return PMX(prob=probability)
        except ImportError:
            try:
                from pymoo.operators.crossover.ox import OrderCrossover
            except ImportError as exc:
                raise ImportError(
                    "pymoo permutation crossover operators are unavailable; upgrade pymoo to a version that ships PMX or OX."
                ) from exc

            class _OrderCrossoverWrapper(OrderCrossover):
                def __init__(self, prob):
                    super().__init__(prob=prob)

            return _OrderCrossoverWrapper(probability)

    crossover = _make_pymoo_perm_crossover(probability=0.9)
    algorithm = NSGA2(
        pop_size=config.population_size,
        sampling=PermutationRandomSampling(),
        crossover=crossover,
        mutation=InversionMutation(prob=mutation_prob),
        eliminate_duplicates=True,
    )
    start = time.perf_counter()
    res = minimize(
        pymoo_problem,
        algorithm,
        ("n_eval", config.max_evaluations),
        seed=config.seed,
        verbose=False,
    )
    end = time.perf_counter()
    total_time_ms = (end - start) * 1000.0
    F = np.asarray(res.F, dtype=float)
    metrics = make_metrics("pymoo_perm_nsga2", "pymoo", total_time_ms, config.max_evaluations, F)
    print_results(metrics)
    logger.info("%s", "=" * 80)
    return metrics
