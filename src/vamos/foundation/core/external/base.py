from __future__ import annotations

import time

import numpy as np

from vamos.foundation.problem.registry import ProblemSelection


class ExternalAlgorithmAdapter:
    """
    Thin adapter to standardize external baseline invocation.
    """

    def __init__(self, name: str, runner_fn):
        self.name = name
        self._runner_fn = runner_fn

    def run(
        self,
        selection: ProblemSelection,
        *,
        use_native_problem: bool,
        config,
        make_metrics,
        print_banner,
        print_results,
    ):
        return self._runner_fn(
            selection,
            use_native_problem=use_native_problem,
            config=config,
            make_metrics=make_metrics,
            print_banner=print_banner,
            print_results=print_results,
        )


def _patch_permutation_swap_mutation_cls(base_cls, permutation_solution_cls, mutation_base_cls):
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

    class _PatchedPermutationSwapMutation(mutation_base_cls):
        def __init__(self, probability: float):
            super().__init__(probability=probability)

        def execute(self, solution):
            import random

            if random.random() <= self.probability:
                perm = solution.variables
                idx_a, idx_b = random.sample(range(len(perm)), 2)
                perm[idx_a], perm[idx_b] = perm[idx_b], perm[idx_a]
                solution.variables = perm
            return solution

        def get_name(self):
            return "Permutation Swap mutation (patched)"

    return _PatchedPermutationSwapMutation


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
        raise ImportError(
            "pymoo is not installed. Install it with 'pip install pymoo' to use this baseline."
        ) from exc

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
    metrics = make_metrics(
        "pymoo_nsga2", "pymoo", total_time_ms, config.max_evaluations, F
    )
    print_results(metrics)
    print("=" * 80)
    return metrics


def _run_jmetalpy_nsga2(
    selection: ProblemSelection,
    *,
    use_native_problem: bool,
    config,
    make_metrics,
    print_banner,
    print_results,
):
    if selection.spec.key != "zdt1":
        raise ValueError("jMetalPy baseline currently supports only ZDT1.")
    problem = selection.instantiate()
    print_banner(problem, selection, "jMetalPy NSGA-II", "jmetalpy")
    try:
        from jmetal.core.problem import FloatProblem
        from jmetal.core.solution import FloatSolution
        from jmetal.algorithm.multiobjective.nsgaii import NSGAII
        from jmetal.operator.crossover import SBXCrossover
        from jmetal.operator.mutation import PolynomialMutation
        from jmetal.problem.multiobjective.zdt import ZDT1
        from jmetal.util.termination_criterion import StoppingByEvaluations
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "jmetalpy is not installed. Install it with 'pip install jmetalpy' to use this baseline."
        ) from exc

    try:
        from jmetal.util.random_generator import PRNG

        PRNG.seed(config.seed)
    except Exception:  # pragma: no cover
        pass

    if use_native_problem:
        jm_problem = ZDT1(number_of_variables=selection.n_var)
    else:

        class _VamosJMetalProblem(FloatProblem):
            def __init__(self, base_problem):
                super().__init__()
                self.base_problem = base_problem
                self.number_of_variables = base_problem.n_var
                self.number_of_objectives = base_problem.n_obj
                self.number_of_constraints = 0
                self.lower_bound = [base_problem.xl] * base_problem.n_var
                self.upper_bound = [base_problem.xu] * base_problem.n_var
                self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
                self.obj_labels = [f"f{i+1}" for i in range(self.number_of_objectives)]

            def evaluate(self, solution: FloatSolution) -> FloatSolution:
                X = np.asarray(solution.variables, dtype=float, copy=False)[np.newaxis, :]
                F = np.empty((1, self.number_of_objectives))
                self.base_problem.evaluate(X, {"F": F})
                solution.objectives = F[0].tolist()
                return solution

            def create_solution(self) -> FloatSolution:
                return FloatSolution(
                    self.lower_bound, self.upper_bound, self.number_of_objectives
                )

        jm_problem = _VamosJMetalProblem(problem)

    mutation = PolynomialMutation(
        probability=1.0 / selection.n_var, distribution_index=20.0
    )
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
    metrics = make_metrics(
        "jmetalpy_nsga2", "jmetalpy", total_time_ms, config.max_evaluations, F
    )
    print_results(metrics)
    print("=" * 80)
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
        raise ImportError(
            "pymoo is not installed. Install it with 'pip install pymoo' to use this baseline."
        ) from exc

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
                    "pymoo permutation crossover operators are unavailable; "
                    "upgrade pymoo to a version that ships PMX or OX."
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
    print("=" * 80)
    return metrics


def _run_jmetalpy_perm_nsga2(
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
        raise ValueError("jMetalPy permutation baseline requires a permutation-encoded problem.")
    print_banner(problem, selection, "jMetalPy NSGA-II (perm)", "jmetalpy")
    try:
        from jmetal.core.operator import Crossover, Mutation
        from jmetal.core.problem import PermutationProblem
        from jmetal.core.solution import PermutationSolution
        from jmetal.algorithm.multiobjective.nsgaii import NSGAII
        from jmetal.util.termination_criterion import StoppingByEvaluations
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "jmetalpy is not installed. Install it with 'pip install jmetalpy' to use this baseline."
        ) from exc

    try:  # pragma: no cover - exercised indirectly when jmetalpy is installed
        from jmetal.operator.mutation import PermutationSwapMutation as _SwapMutationCandidate
    except ImportError:  # pragma: no cover
        try:
            from jmetal.operator.mutation import SwapMutation as _SwapMutationCandidate
        except ImportError as exc:
            raise ImportError(
                "The installed jmetalpy version does not expose a permutation swap mutation "
                "(requires PermutationSwapMutation>=1.9 or SwapMutation<=1.7)."
            ) from exc

    _SwapMutationOp = _patch_permutation_swap_mutation_cls(
        _SwapMutationCandidate, PermutationSolution, Mutation
    )

    try:
        from vamos.operators.permutation import order_crossover as _vamos_order_crossover
    except ImportError as exc:  # pragma: no cover
        raise ImportError("VAMOS permutation operators are unavailable.") from exc

    class _VamosOrderCrossover(Crossover):
        def __init__(self, probability: float, seed: int):
            super().__init__(probability=probability)
            self._rng = np.random.default_rng(seed)

        def execute(self, parents):
            if len(parents) != 2:
                raise Exception(f"Expected 2 parents, received {len(parents)}.")
            parent_arrays = np.asarray(
                [parents[0].variables, parents[1].variables], dtype=int
            )
            children_arrays = _vamos_order_crossover(
                parent_arrays, self.probability, self._rng
            )
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

    class _VamosJMetalPermutationProblem(PermutationProblem):
        def __init__(self, base_problem):
            super().__init__()
            self.base_problem = base_problem
            self._n_var = int(base_problem.n_var)
            self._n_obj = int(base_problem.n_obj)
            self._n_con = 0
            self._name = f"VAMOS-{base_problem.__class__.__name__}"
            self.obj_directions = [self.MINIMIZE] * self._n_obj
            self.obj_labels = [f"f{i+1}" for i in range(self._n_obj)]

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
    print("=" * 80)
    return metrics


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
        raise ImportError(
            "pygmo is not installed. Install it with 'pip install pygmo' to use this baseline."
        ) from exc

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
    print("=" * 80)
    return metrics


EXTERNAL_ALGORITHM_RUNNERS = {
    "pymoo_nsga2": _run_pymoo_nsga2,
    "jmetalpy_nsga2": _run_jmetalpy_nsga2,
    "pygmo_nsga2": _run_pygmo_nsga2,
    "pymoo_perm_nsga2": _run_pymoo_perm_nsga2,
    "jmetalpy_perm_nsga2": _run_jmetalpy_perm_nsga2,
}

EXTERNAL_ALGORITHM_ADAPTERS = {
    name: ExternalAlgorithmAdapter(name, fn) for name, fn in EXTERNAL_ALGORITHM_RUNNERS.items()
}


def resolve_external_algorithm(name: str) -> ExternalAlgorithmAdapter:
    try:
        return EXTERNAL_ALGORITHM_ADAPTERS[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        available = ", ".join(sorted(EXTERNAL_ALGORITHM_ADAPTERS))
        raise ValueError(f"Unknown external algorithm '{name}'. Available: {available}") from exc


def run_external(
    name: str,
    selection: ProblemSelection,
    *,
    use_native_problem: bool,
    config,
    make_metrics,
    print_banner,
    print_results,
):
    adapter = resolve_external_algorithm(name)
    try:
        return adapter.run(
            selection,
            use_native_problem=use_native_problem,
            config=config,
            make_metrics=make_metrics,
            print_banner=print_banner,
            print_results=print_results,
        )
    except ImportError as exc:
        print(f"Skipping {name}: {exc}")
        print("=" * 80)
        return None
