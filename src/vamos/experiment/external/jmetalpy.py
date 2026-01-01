from __future__ import annotations

import logging
import time
from typing import Any, Callable, Sequence

import numpy as np

from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import ProblemSelection

logger = logging.getLogger(__name__)
MetricsBuilder = Callable[[str, str, float, int, np.ndarray], dict[str, Any]]
Printer = Callable[..., Any]


def _patch_permutation_swap_mutation_cls(
    base_cls: type[Any],
    permutation_solution_cls: type[Any],
    mutation_base_cls: type[Any],
) -> type[Any]:
    """
    Ensure the provided swap mutation actually updates solutions.
    Some jMetalPy releases expose a swap operator that operates on copies
    returned by `PermutationSolution.variables` without writing back,
    which becomes a no-op. We detect this behaviour and inject a small
    patched implementation that mutates the permutation and reassigns it.
    """
    try:
        test_sol = permutation_solution_cls(5, 1)
    except TypeError:
        test_sol = permutation_solution_cls(5, 1, 0)
    baseline_perm = list(range(5))
    test_sol.variables = baseline_perm
    try:
        base_cls(probability=1.0).execute(test_sol)
    except Exception:
        needs_patch = True
    else:
        needs_patch = test_sol.variables == baseline_perm

    if not needs_patch:
        return base_cls

    class _PatchedPermutationSwapMutation(mutation_base_cls):  # type: ignore[misc]
        def __init__(self, probability: float) -> None:
            super().__init__(probability=probability)

        def execute(self, solution: Any) -> Any:
            import random

            if random.random() <= self.probability:
                perm = solution.variables
                idx_a, idx_b = random.sample(range(len(perm)), 2)
                perm[idx_a], perm[idx_b] = perm[idx_b], perm[idx_a]
                solution.variables = perm
            return solution

        def get_name(self) -> str:
            return "Permutation Swap mutation (patched)"

    return _PatchedPermutationSwapMutation


def _run_jmetalpy_nsga2(
    selection: ProblemSelection,
    *,
    use_native_problem: bool,
    config: ExperimentConfig,
    make_metrics: MetricsBuilder,
    print_banner: Printer,
    print_results: Printer,
) -> dict[str, Any]:
    if selection.spec.key != "zdt1":
        raise ValueError("jMetalPy baseline currently supports only ZDT1.")
    problem = selection.instantiate()
    print_banner(problem, selection, "jMetalPy NSGA-II", "jmetalpy")
    try:
        from jmetal.core.problem import FloatProblem
        from jmetal.core.solution import FloatSolution
        from jmetal.algorithm.multiobjective.nsgaii import (
            NSGAII,
        )
        from jmetal.operator.crossover import SBXCrossover
        from jmetal.operator.mutation import (
            PolynomialMutation,
        )
        from jmetal.problem.multiobjective.zdt import ZDT1
        from jmetal.util.termination_criterion import (
            StoppingByEvaluations,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError("jmetalpy is not installed. Install it with 'pip install jmetalpy' to use this baseline.") from exc

    try:
        from jmetal.util.random_generator import PRNG

        PRNG.seed(config.seed)
    except Exception:  # pragma: no cover
        pass

    if use_native_problem:
        jm_problem = ZDT1(number_of_variables=selection.n_var)
    else:

        class _VamosJMetalProblem(FloatProblem):  # type: ignore[misc]
            def __init__(self, base_problem: Any) -> None:
                super().__init__()
                self.base_problem = base_problem
                self.number_of_variables = base_problem.n_var
                self.number_of_objectives = base_problem.n_obj
                self.number_of_constraints = 0
                self.lower_bound = [base_problem.xl] * base_problem.n_var
                self.upper_bound = [base_problem.xu] * base_problem.n_var
                self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
                self.obj_labels = [f"f{i + 1}" for i in range(self.number_of_objectives)]

            def evaluate(self, solution: FloatSolution) -> FloatSolution:
                X = np.asarray(solution.variables, dtype=float, copy=False)[np.newaxis, :]
                F = np.empty((1, self.number_of_objectives))
                self.base_problem.evaluate(X, {"F": F})
                solution.objectives = F[0].tolist()
                return solution

            def create_solution(self) -> FloatSolution:
                return FloatSolution(self.lower_bound, self.upper_bound, self.number_of_objectives)

        jm_problem = _VamosJMetalProblem(problem)

    mutation = PolynomialMutation(probability=1.0 / selection.n_var, distribution_index=20.0)
    crossover = SBXCrossover(probability=0.9, distribution_index=20.0)
    algorithm = NSGAII(
        problem=jm_problem,
        population_size=config.population_size,
        offspring_population_size=config.population_size,
        mutation=mutation,
        crossover=crossover,
        termination_criterion=StoppingByEvaluations(max_evaluations=config.max_evaluations),
    )
    start = time.perf_counter()
    algorithm.run()
    end = time.perf_counter()
    total_time_ms = (end - start) * 1000.0
    get_result_fn = getattr(algorithm, "result", None)
    solutions = get_result_fn() if callable(get_result_fn) else []
    F = np.array([sol.objectives for sol in solutions], dtype=float)
    metrics = make_metrics("jmetalpy_nsga2", "jmetalpy", total_time_ms, config.max_evaluations, F)
    print_results(metrics)
    logger.info("%s", "=" * 80)
    return metrics


def _run_jmetalpy_perm_nsga2(
    selection: ProblemSelection,
    *,
    use_native_problem: bool,
    config: ExperimentConfig,
    make_metrics: MetricsBuilder,
    print_banner: Printer,
    print_results: Printer,
) -> dict[str, Any]:
    problem = selection.instantiate()
    encoding = getattr(problem, "encoding", "continuous")
    if encoding != "permutation":
        raise ValueError("jMetalPy permutation baseline requires a permutation-encoded problem.")
    print_banner(problem, selection, "jMetalPy NSGA-II (perm)", "jmetalpy")
    try:
        from jmetal.core.operator import Crossover, Mutation
        from jmetal.core.problem import PermutationProblem
        from jmetal.core.solution import PermutationSolution
        from jmetal.algorithm.multiobjective.nsgaii import NSGAII
        from jmetal.util.termination_criterion import StoppingByEvaluations
    except ImportError as exc:  # pragma: no cover
        raise ImportError("jmetalpy is not installed. Install it with 'pip install jmetalpy' to use this baseline.") from exc

    try:  # pragma: no cover - exercised indirectly when jmetalpy is installed
        from jmetal.operator.mutation import (
            PermutationSwapMutation as _SwapMutationCandidate,
        )
    except ImportError:  # pragma: no cover
        try:
            from jmetal.operator.mutation import (
                SwapMutation as _SwapMutationCandidate,
            )
        except ImportError as exc:
            raise ImportError(
                "The installed jmetalpy version does not expose a permutation swap mutation "
                "(requires PermutationSwapMutation>=1.9 or SwapMutation<=1.7)."
            ) from exc

    _SwapMutationOp = _patch_permutation_swap_mutation_cls(_SwapMutationCandidate, PermutationSolution, Mutation)

    try:
        from vamos.operators.permutation import order_crossover as _vamos_order_crossover
    except ImportError as exc:  # pragma: no cover
        raise ImportError("VAMOS permutation operators are unavailable.") from exc

    class _VamosOrderCrossover(Crossover):  # type: ignore[misc]
        def __init__(self, probability: float, seed: int) -> None:
            super().__init__(probability=probability)
            self._rng = np.random.default_rng(seed)

        def execute(self, parents: Sequence[Any]) -> list[Any]:
            if len(parents) != 2:
                raise Exception(f"Expected 2 parents, received {len(parents)}.")
            parent_arrays = np.asarray([parents[0].variables, parents[1].variables], dtype=int)
            children_arrays = _vamos_order_crossover(parent_arrays, self.probability, self._rng)
            offspring = [parents[0].__copy__(), parents[1].__copy__()]
            for child_sol, arr in zip(offspring, children_arrays, strict=False):
                child_sol.variables = arr.tolist()
            return offspring

        def get_number_of_parents(self) -> int:
            return 2

        def get_number_of_children(self) -> int:
            return 2

        def get_name(self) -> str:
            return "Order crossover (VAMOS)"

    class _VamosJMetalPermutationProblem(PermutationProblem):  # type: ignore[misc]
        def __init__(self, base_problem: Any) -> None:
            super().__init__()
            self.base_problem = base_problem
            self._n_var = int(base_problem.n_var)
            self._n_obj = int(base_problem.n_obj)
            self._n_con = 0
            self._name = f"VAMOS-{base_problem.__class__.__name__}"
            self.obj_directions = [self.MINIMIZE] * self._n_obj
            self.obj_labels = [f"f{i + 1}" for i in range(self._n_obj)]

        def number_of_variables(self) -> int:
            return self._n_var

        def number_of_objectives(self) -> int:
            return self._n_obj

        def number_of_constraints(self) -> int:
            return self._n_con

        def name(self) -> str:
            return self._name

        def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
            perm = np.asarray(solution.variables, dtype=int)[np.newaxis, :]
            F = np.empty((1, self._n_obj))
            self.base_problem.evaluate(perm, {"F": F})
            solution.objectives = F[0].tolist()
            return solution

        def create_solution(self) -> PermutationSolution:
            sol = PermutationSolution(self._n_var, self._n_obj, self._n_con)
            sol.variables = np.random.permutation(self._n_var).tolist()
            return sol

    jm_problem = _VamosJMetalPermutationProblem(problem)
    mutation_prob = min(1.0, 2.0 / max(1, problem.n_var))
    crossover = _VamosOrderCrossover(probability=0.9, seed=config.seed)
    mutation = _SwapMutationOp(probability=mutation_prob)
    algorithm = NSGAII(
        problem=jm_problem,
        population_size=config.population_size,
        offspring_population_size=config.population_size,
        mutation=mutation,
        crossover=crossover,
        termination_criterion=StoppingByEvaluations(max_evaluations=config.max_evaluations),
    )
    start = time.perf_counter()
    algorithm.run()
    end = time.perf_counter()
    total_time_ms = (end - start) * 1000.0
    get_result_fn = getattr(algorithm, "result", None)
    solutions = get_result_fn() if callable(get_result_fn) else []
    F = np.array([sol.objectives for sol in solutions], dtype=float)
    metrics = make_metrics("jmetalpy_perm_nsga2", "jmetalpy", total_time_ms, config.max_evaluations, F)
    print_results(metrics)
    logger.info("%s", "=" * 80)
    return metrics
