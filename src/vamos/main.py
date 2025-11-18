# main.py
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
import numpy as np

# Allow running via `python src/vamos/main.py` without installing the package.
if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from vamos.algorithm.config import MOEADConfig, NSGAIIConfig, NSGAIIIConfig, SMSEMOAConfig
from vamos.algorithm.hypervolume import hypervolume
from vamos.algorithm.moead import MOEAD
from vamos.algorithm.nsgaii import NSGAII
from vamos.algorithm.nsga3 import NSGAIII
from vamos.algorithm.smsemoa import SMSEMOA
from vamos.kernel.numpy_backend import NumPyKernel
from vamos.problem.registry import available_problem_names, make_problem_selection, ProblemSelection


POPULATION_SIZE = 100
MAX_EVALUATIONS = 25000
SEED = 42
DECIMAL_PRECISION = 6
OUTPUT_ROOT = "results"
TITLE = "VAMOS: Vectorized Architecture for Multiobjective Optimization Studies"
DEFAULT_ALGORITHM = "nsgaii"
DEFAULT_ENGINE = "numpy"
DEFAULT_PROBLEM = "zdt1"
ENABLED_ALGORITHMS = ("nsgaii", "moead", "smsemoa")
OPTIONAL_ALGORITHMS = ("nsga3",)
EXTERNAL_ALGORITHM_NAMES = ("pymoo_nsga2", "jmetalpy_nsga2", "pygmo_nsga2")
HV_REFERENCE_OFFSET = 0.1
EXPERIMENT_BACKENDS = (
    "numpy",
    "numba",
    "moocore",
    "moocore_v2",
)
PROBLEM_SET_PRESETS = {
    "families": (
        {"problem": "zdt1"},
        {"problem": "dtlz2"},
        {"problem": "wfg4"},
        {"problem": "tsp6"},
    ),
    "tsplib_kro100": tuple({"problem": key} for key in ("kroa100", "krob100", "kroc100", "krod100", "kroe100")),
    "all": tuple({"problem": name} for name in available_problem_names()),
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Vectorized multi-objective optimization demo across benchmark problems."
    )
    parser.add_argument(
        "--algorithm",
        choices=(
            *ENABLED_ALGORITHMS,
            *OPTIONAL_ALGORITHMS,
            *EXTERNAL_ALGORITHM_NAMES,
            "both",
        ),
        default=DEFAULT_ALGORITHM,
        help=(
            "Algorithm to run (use 'both' to execute the default internal algorithms sequentially; "
            "combine with --include-external to add third-party baselines)."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=("numpy", "numba", "moocore", "moocore_v2"),
        default=DEFAULT_ENGINE,
        help="Kernel backend to use (default: numpy).",
    )
    parser.add_argument(
        "--problem",
        choices=available_problem_names(),
        default=DEFAULT_PROBLEM,
        help="Benchmark problem to solve.",
    )
    if PROBLEM_SET_PRESETS:
        parser.add_argument(
            "--problem-set",
            choices=tuple(PROBLEM_SET_PRESETS.keys()),
            help=(
                "Run a predefined set of benchmark problems sequentially "
                "(e.g., 'families' runs ZDT1, DTLZ2, and WFG4). Overrides --problem."
            ),
        )
    parser.add_argument(
        "--n-var",
        type=int,
        help="Override the number of decision variables for the selected problem.",
    )
    parser.add_argument(
        "--n-obj",
        type=int,
        help="Override the number of objectives (if the problem supports it).",
    )
    parser.add_argument(
        "--experiment",
        choices=("backends",),
        help="Run a predefined experiment (e.g., compare all backends).",
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Include PyMOO/jMetalPy/PyGMO baselines when running algorithms.",
    )
    parser.add_argument(
        "--external-problem-source",
        choices=("native", "vamos"),
        default="native",
        help=(
            "For external baselines, choose whether to use each library's native benchmark "
            "implementation ('native') or wrap the VAMOS problem definition ('vamos')."
        ),
    )
    return parser.parse_args()


def _resolve_kernel(engine_name: str):
    if engine_name == "numpy":
        return NumPyKernel()
    if engine_name == "numba":
        try:
            from vamos.kernel.numba_backend import NumbaKernel
        except ImportError as exc:
            raise SystemExit(
                "El backend 'numba' requiere la dependencia numba instalada.\n"
                f"Error original: {exc}"
            ) from exc
        return NumbaKernel()
    if engine_name == "moocore":
        try:
            from vamos.kernel.moocore_backend import MooCoreKernel
        except ImportError as exc:
            raise SystemExit(
                "El backend 'moocore' requiere la dependencia moocore instalada.\n"
                f"Error original: {exc}"
            ) from exc
        return MooCoreKernel()
    if engine_name == "moocore_v2":
        try:
            from vamos.kernel.moocore_backend import MooCoreKernelV2
        except ImportError as exc:
            raise SystemExit(
                "El backend 'moocore_v2' requiere la dependencia moocore instalada.\n"
                f"Error original: {exc}"
            ) from exc
        return MooCoreKernelV2()
    raise ValueError(f"Engine desconocido: {engine_name}")


def _default_weight_path(problem_name: str, n_obj: int, pop_size: int) -> str:
    safe_name = problem_name.lower()
    filename = f"{safe_name}_{n_obj}obj_pop{pop_size}.csv"
    return os.path.join("build", "weights", filename)


def _compute_hv_reference(fronts: list[np.ndarray]) -> np.ndarray:
    stacked = np.vstack(fronts)
    return stacked.max(axis=0) + HV_REFERENCE_OFFSET


def _problem_output_dir(selection: ProblemSelection) -> str:
    safe = selection.spec.label.replace(" ", "_").upper()
    return os.path.join(OUTPUT_ROOT, f"{safe}")


def _run_output_dir(
    selection: ProblemSelection, algorithm_name: str, engine_name: str, seed: int
) -> str:
    base = _problem_output_dir(selection)
    return os.path.join(
        base,
        algorithm_name.lower(),
        engine_name.lower(),
        f"seed_{seed}",
    )


def _resolve_problem_selection(args) -> ProblemSelection:
    try:
        return make_problem_selection(
            args.problem, n_var=args.n_var, n_obj=args.n_obj
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _resolve_problem_selections(args) -> list[ProblemSelection]:
    if not getattr(args, "problem_set", None):
        return [_resolve_problem_selection(args)]
    if args.n_var is not None or args.n_obj is not None:
        raise SystemExit("--n-var/--n-obj overrides cannot be combined with --problem-set.")
    presets = PROBLEM_SET_PRESETS.get(args.problem_set, ())
    if not presets:
        raise SystemExit(f"No problem set preset named '{args.problem_set}'.")
    selections = []
    for entry in presets:
        selection = make_problem_selection(
            entry["problem"],
            n_var=entry.get("n_var"),
            n_obj=entry.get("n_obj"),
        )
        selections.append(selection)
    return selections


def _build_algorithm(algorithm_name: str, engine_name: str, problem):
    kernel_backend = _resolve_kernel(engine_name)
    encoding = getattr(problem, "encoding", "continuous")
    if encoding == "permutation" and algorithm_name != "nsgaii":
        raise ValueError(
            f"Problem '{problem.__class__.__name__}' uses permutation encoding; "
            f"currently only NSGA-II supports this representation."
        )
    if algorithm_name == "nsgaii":
        if encoding == "permutation":
            crossover_cfg = ("ox", {"prob": 0.9})
            mutation_cfg = ("swap", {"prob": "2/n"})
        else:
            crossover_cfg = ("sbx", {"prob": 0.9, "eta": 20.0})
            mutation_cfg = ("pm", {"prob": "1/n", "eta": 20.0})
        cfg_builder = (
            NSGAIIConfig()
            .pop_size(POPULATION_SIZE)
            .crossover(crossover_cfg[0], **crossover_cfg[1])
            .mutation(mutation_cfg[0], **mutation_cfg[1])
            .selection("tournament", pressure=2)
            .survival("nsga2")
            .engine(engine_name)
        )
        cfg = cfg_builder.fixed()
        return NSGAII(cfg.to_dict(), kernel=kernel_backend), cfg

    if algorithm_name == "moead":
        weight_path = _default_weight_path(
            problem.__class__.__name__, problem.n_obj, POPULATION_SIZE
        )
        cfg_builder = (
            MOEADConfig()
            .pop_size(POPULATION_SIZE)
            .neighbor_size(min(20, POPULATION_SIZE))
            .delta(0.9)
            .replace_limit(2)
            .crossover("sbx", prob=0.9, eta=20.0)
            .mutation("pm", prob="1/n", eta=20.0)
            .aggregation("tchebycheff")
            .weight_vectors(path=weight_path)
            .engine(engine_name)
        )
        cfg = cfg_builder.fixed()
        return MOEAD(cfg.to_dict(), kernel=kernel_backend), cfg

    if algorithm_name == "smsemoa":
        cfg_builder = (
            SMSEMOAConfig()
            .pop_size(POPULATION_SIZE)
            .crossover("sbx", prob=0.9, eta=20.0)
            .mutation("pm", prob="1/n", eta=20.0)
            .selection("tournament", pressure=2)
            .reference_point(offset=0.1, adaptive=True)
            .engine(engine_name)
        )
        cfg = cfg_builder.fixed()
        return SMSEMOA(cfg.to_dict(), kernel=kernel_backend), cfg

    if algorithm_name == "nsga3":
        ref_path = _default_weight_path(
            problem.__class__.__name__, problem.n_obj, POPULATION_SIZE
        )
        cfg_builder = (
            NSGAIIIConfig()
            .pop_size(POPULATION_SIZE)
            .crossover("sbx", prob=0.9, eta=20.0)
            .mutation("pm", prob="1/n", eta=20.0)
            .selection("tournament", pressure=2)
            .reference_directions(path=ref_path)
            .engine(engine_name)
        )
        cfg = cfg_builder.fixed()
        return NSGAIII(cfg.to_dict(), kernel=kernel_backend), cfg

    raise ValueError(f"Unsupported algorithm '{algorithm_name}'.")


def _print_run_banner(
    problem, problem_selection: ProblemSelection, algorithm_label: str, backend_label: str
):
    print("=" * 80)
    print(TITLE)
    print("=" * 80)
    print(f"Problem: {problem_selection.spec.label}")
    if problem_selection.spec.description:
        print(f"Description: {problem_selection.spec.description}")
    print(f"Decision variables: {problem.n_var}")
    print(f"Objectives: {problem.n_obj}")
    encoding = getattr(problem, "encoding", problem_selection.spec.encoding)
    if encoding:
        print(f"Encoding: {encoding}")
    print(f"Algorithm: {algorithm_label}")
    print(f"Backend: {backend_label}")
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Max evaluations: {MAX_EVALUATIONS}")
    print("-" * 80)


def _make_metrics(
    algorithm_name: str,
    engine_name: str,
    total_time_ms: float,
    evaluations: int,
    F: np.ndarray,
):
    F = np.asarray(F, dtype=float)
    evals_per_sec = evaluations / max(total_time_ms / 1000.0, 1e-9)
    spread = None
    if F.ndim == 2 and F.shape[0] > 0:
        ranges = F.max(axis=0) - F.min(axis=0)
        spread = float(np.linalg.norm(ranges))
    return {
        "algorithm": algorithm_name,
        "engine": engine_name,
        "time_ms": total_time_ms,
        "evaluations": evaluations,
        "evals_per_sec": evals_per_sec,
        "spread": spread,
        "F": F,
    }


def _print_run_results(metrics: dict):
    F = metrics["F"]
    print("Algorithm finished")
    print("-" * 80)
    print("PERFORMANCE RESULTS:")
    print(f"Total time: {metrics['time_ms']:.2f} ms")
    print(f"Evaluations: {metrics['evaluations']}")
    print(f"Evaluations/second: {metrics['evals_per_sec']:.0f}")
    print(f"Final solutions: {F.shape[0]}")
    print("\nSOLUTION QUALITY:")
    obj_min = F.min(axis=0)
    obj_max = F.max(axis=0)
    for i, (mn, mx) in enumerate(zip(obj_min, obj_max), start=1):
        print(
            f"  Objective {i} range: "
            f"[{mn:.{DECIMAL_PRECISION}f}, {mx:.{DECIMAL_PRECISION}f}]"
        )
    spread = metrics["spread"]
    if spread is not None:
        print(f"  Approximate front spread in f1: {spread:.{DECIMAL_PRECISION}f}")


def _build_run_metadata(
    selection: ProblemSelection,
    algorithm_name: str,
    engine_name: str,
    cfg_data,
    metrics: dict,
    kernel_backend=None,
    seed: int = SEED,
) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()
    config_payload = cfg_data.to_dict() if cfg_data is not None else None
    problem_info = {
        "key": selection.spec.key,
        "label": selection.spec.label,
        "description": selection.spec.description,
        "n_var": selection.n_var,
        "n_obj": selection.n_obj,
        "encoding": selection.spec.encoding,
    }
    backend_info = {
        "name": engine_name,
        "device": kernel_backend.device() if kernel_backend else "external",
        "capabilities": sorted(set(kernel_backend.capabilities())) if kernel_backend else [],
        "quality_indicators": sorted(set(kernel_backend.quality_indicators()))
        if kernel_backend
        else [],
    }
    metric_payload = {
        "time_ms": metrics["time_ms"],
        "evaluations": metrics["evaluations"],
        "evals_per_sec": metrics["evals_per_sec"],
        "spread": metrics["spread"],
    }
    return {
        "title": TITLE,
        "timestamp": timestamp,
        "algorithm": algorithm_name,
        "backend": engine_name,
        "backend_info": backend_info,
        "seed": seed,
        "population_size": POPULATION_SIZE,
        "max_evaluations": MAX_EVALUATIONS,
        "problem": problem_info,
        "config": config_payload,
        "metrics": metric_payload,
    }


def _run_single(
    engine_name: str, algorithm_name: str, selection: ProblemSelection, seed: int
):
    problem = selection.instantiate()
    display_algo = algorithm_name.upper()
    _print_run_banner(problem, selection, display_algo, engine_name)
    algorithm, cfg_data = _build_algorithm(algorithm_name, engine_name, problem)
    kernel_backend = algorithm.kernel

    start = time.perf_counter()
    result = algorithm.run(problem, termination=("n_eval", MAX_EVALUATIONS), seed=seed)
    end = time.perf_counter()

    total_time_ms = (end - start) * 1000.0
    F = result["F"]

    metrics = _make_metrics(
        algorithm_name, engine_name, total_time_ms, MAX_EVALUATIONS, F
    )
    metrics["config"] = cfg_data
    if kernel_backend is not None:
        metrics["_kernel_backend"] = kernel_backend
        metrics["backend_device"] = kernel_backend.device()
        metrics["backend_capabilities"] = sorted(set(kernel_backend.capabilities()))
    else:
        metrics["backend_device"] = "external"
        metrics["backend_capabilities"] = []
    _print_run_results(metrics)
    output_dir = _run_output_dir(selection, algorithm_name, engine_name, SEED)
    metrics["output_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)
    fun_path = os.path.join(output_dir, "FUN.csv")
    np.savetxt(fun_path, F, delimiter=",")
    time_path = os.path.join(output_dir, "time.txt")
    with open(time_path, "w", encoding="utf-8") as f:
        f.write(f"{total_time_ms:.2f}\n")
    metadata = _build_run_metadata(
        selection,
        algorithm_name,
        engine_name,
        cfg_data,
        metrics,
        kernel_backend=kernel_backend,
        seed=seed,
    )
    metadata["artifacts"] = {"fun": "FUN.csv", "time_ms": "time.txt"}
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print("\nResults stored in:", output_dir)
    print("=" * 80)

    return metrics


def _run_pymoo_nsga2(selection: ProblemSelection, *, use_native_problem: bool):
    if selection.spec.key != "zdt1":
        raise ValueError("PyMOO baseline currently supports only ZDT1.")
    problem = selection.instantiate()
    _print_run_banner(problem, selection, "PyMOO NSGA-II", "pymoo")
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
        pop_size=POPULATION_SIZE,
        sampling=FloatRandomSampling(),
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )
    start = time.perf_counter()
    res = minimize(
        pymoo_problem,
        algorithm,
        ("n_eval", MAX_EVALUATIONS),
        seed=SEED,
        verbose=False,
    )
    end = time.perf_counter()
    total_time_ms = (end - start) * 1000.0
    F = np.asarray(res.F, dtype=float)
    metrics = _make_metrics(
        "pymoo_nsga2", "pymoo", total_time_ms, MAX_EVALUATIONS, F
    )
    _print_run_results(metrics)
    print("=" * 80)
    return metrics


def _run_jmetalpy_nsga2(selection: ProblemSelection, *, use_native_problem: bool):
    if selection.spec.key != "zdt1":
        raise ValueError("jMetalPy baseline currently supports only ZDT1.")
    problem = selection.instantiate()
    _print_run_banner(problem, selection, "jMetalPy NSGA-II", "jmetalpy")
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

        PRNG.seed(SEED)
    except Exception:  # pragma: no cover - older versions may not expose PRNG
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
        population_size=POPULATION_SIZE,
        offspring_population_size=POPULATION_SIZE,
        mutation=mutation,
        crossover=crossover,
        termination_criterion=StoppingByEvaluations(max_evaluations=MAX_EVALUATIONS),
    )
    start = time.perf_counter()
    algorithm.run()
    end = time.perf_counter()
    total_time_ms = (end - start) * 1000.0
    get_result_fn = getattr(algorithm, "result", None)
    solutions = get_result_fn() if callable(get_result_fn) else []
    F = np.array([sol.objectives for sol in solutions], dtype=float)
    metrics = _make_metrics(
        "jmetalpy_nsga2", "jmetalpy", total_time_ms, MAX_EVALUATIONS, F
    )
    _print_run_results(metrics)
    print("=" * 80)
    return metrics


def _run_pymoo_perm_nsga2(selection: ProblemSelection, *, use_native_problem: bool):
    problem = selection.instantiate()
    encoding = getattr(problem, "encoding", "continuous")
    if encoding != "permutation":
        raise ValueError("PyMOO permutation baseline requires a permutation-encoded problem.")
    _print_run_banner(problem, selection, "PyMOO NSGA-II (perm)", "pymoo")
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.operators.sampling.rnd import PermutationRandomSampling
        from pymoo.operators.crossover.pmx import PMX
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
    algorithm = NSGA2(
        pop_size=POPULATION_SIZE,
        sampling=PermutationRandomSampling(),
        crossover=PMX(prob=0.9),
        mutation=InversionMutation(prob=mutation_prob),
        eliminate_duplicates=True,
    )
    start = time.perf_counter()
    res = minimize(
        pymoo_problem,
        algorithm,
        ("n_eval", MAX_EVALUATIONS),
        seed=SEED,
        verbose=False,
    )
    end = time.perf_counter()
    total_time_ms = (end - start) * 1000.0
    F = np.asarray(res.F, dtype=float)
    metrics = _make_metrics("pymoo_perm_nsga2", "pymoo", total_time_ms, MAX_EVALUATIONS, F)
    _print_run_results(metrics)
    print("=" * 80)
    return metrics


def _run_jmetalpy_perm_nsga2(selection: ProblemSelection, *, use_native_problem: bool):
    problem = selection.instantiate()
    encoding = getattr(problem, "encoding", "continuous")
    if encoding != "permutation":
        raise ValueError("jMetalPy permutation baseline requires a permutation-encoded problem.")
    _print_run_banner(problem, selection, "jMetalPy NSGA-II (perm)", "jmetalpy")
    try:
        from jmetal.core.problem import PermutationProblem
        from jmetal.core.solution import PermutationSolution
        from jmetal.algorithm.multiobjective.nsgaii import NSGAII
        from jmetal.operator.crossover import PMXCrossover
        from jmetal.operator.mutation import SwapMutation
        from jmetal.util.termination_criterion import StoppingByEvaluations
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "jmetalpy is not installed. Install it with 'pip install jmetalpy' to use this baseline."
        ) from exc

    class _VamosJMetalPermutationProblem(PermutationProblem):
        def __init__(self, base_problem):
            super().__init__()
            self.base_problem = base_problem
            self.number_of_variables = base_problem.n_var
            self.number_of_objectives = base_problem.n_obj
            self.number_of_constraints = 0
            self.obj_directions = [self.MINIMIZE] * self.number_of_objectives
            self.obj_labels = [f"f{i+1}" for i in range(self.number_of_objectives)]

        def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
            perm = np.asarray(solution.variables, dtype=int)[np.newaxis, :]
            F = np.empty((1, self.number_of_objectives))
            self.base_problem.evaluate(perm, {"F": F})
            solution.objectives = F[0].tolist()
            return solution

        def create_solution(self) -> PermutationSolution:
            sol = PermutationSolution(self.number_of_variables, self.number_of_objectives, self.number_of_constraints)
            sol.variables = np.random.permutation(self.number_of_variables).tolist()
            return sol

    jm_problem = _VamosJMetalPermutationProblem(problem)
    mutation_prob = min(1.0, 2.0 / max(1, problem.n_var))
    crossover = PMXCrossover(probability=0.9)
    mutation = SwapMutation(probability=mutation_prob)
    algorithm = NSGAII(
        problem=jm_problem,
        population_size=POPULATION_SIZE,
        offspring_population_size=POPULATION_SIZE,
        mutation=mutation,
        crossover=crossover,
        termination_criterion=StoppingByEvaluations(max_evaluations=MAX_EVALUATIONS),
    )
    start = time.perf_counter()
    algorithm.run()
    end = time.perf_counter()
    total_time_ms = (end - start) * 1000.0
    solutions = algorithm.get_result() if hasattr(algorithm, "get_result") else []
    if isinstance(solutions, PermutationSolution):
        solutions = [solutions]
    F = np.array([sol.objectives for sol in solutions], dtype=float) if solutions else np.empty((0, problem.n_obj))
    metrics = _make_metrics("jmetalpy_perm_nsga2", "jmetalpy", total_time_ms, MAX_EVALUATIONS, F)
    _print_run_results(metrics)
    print("=" * 80)
    return metrics


def _run_pygmo_nsga2(selection: ProblemSelection, *, use_native_problem: bool):
    if selection.spec.key != "zdt1":
        raise ValueError("PyGMO baseline currently supports only ZDT1.")
    problem = selection.instantiate()
    _print_run_banner(problem, selection, "PyGMO NSGA-II", "pygmo")
    try:
        import pygmo as pg
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pygmo is not installed. Install it (e.g., via conda) to use this baseline."
        ) from exc

    generations = max(1, (MAX_EVALUATIONS - POPULATION_SIZE) // POPULATION_SIZE)
    uda = pg.nsga2(
        gen=generations,
        seed=SEED,
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
    pop = pg.population(pg_problem, size=POPULATION_SIZE, seed=SEED)
    start = time.perf_counter()
    pop = algo.evolve(pop)
    end = time.perf_counter()
    total_eval = POPULATION_SIZE * (generations + 1)
    total_time_ms = (end - start) * 1000.0
    F = np.asarray(pop.get_f(), dtype=float)
    metrics = _make_metrics("pygmo_nsga2", "pygmo", total_time_ms, total_eval, F)
    _print_run_results(metrics)
    print("=" * 80)
    return metrics


EXTERNAL_ALGORITHM_RUNNERS = {
    "pymoo_nsga2": _run_pymoo_nsga2,
    "jmetalpy_nsga2": _run_jmetalpy_nsga2,
    "pygmo_nsga2": _run_pygmo_nsga2,
    "pymoo_perm_nsga2": _run_pymoo_perm_nsga2,
    "jmetalpy_perm_nsga2": _run_jmetalpy_perm_nsga2,
}


def _execute_external_runner(
    name: str, selection: ProblemSelection, *, use_native_problem: bool
):
    runner = EXTERNAL_ALGORITHM_RUNNERS[name]
    try:
        return runner(selection, use_native_problem=use_native_problem)
    except ImportError as exc:
        print(f"Skipping {name}: {exc}")
        print("=" * 80)
        return None


def _print_summary(results, hv_ref_point: np.ndarray):
    print("\nExperiment summary")
    print("-" * 80)
    header = (
        f"{'Algo':<12} {'Backend':<10} {'Time (ms)':>12} {'Eval/s':>12} {'HV':>12} {'Spread f1':>12}"
    )
    print(header)
    print("-" * len(header))
    for res in results:
        spread = res["spread"]
        spread_txt = f"{spread:.6f}" if spread is not None else "-"
        print(
            f"{res['algorithm']:<12} "
            f"{res['engine']:<10} "
            f"{res['time_ms']:>12.2f} "
            f"{res['evals_per_sec']:>12.0f} "
            f"{res['hv']:>12.6f} "
            f"{spread_txt:>12}"
        )
    print("-" * len(header))
    ref_txt = np.array2string(hv_ref_point, precision=3, suppress_small=True)
    print(f"Hypervolume reference point: {ref_txt}")


def _non_dominated_mask(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.array([], dtype=bool)
    n_points = points.shape[0]
    mask = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not mask[i]:
            continue
        dominates = np.all(points[i] <= points, axis=1) & np.any(points[i] < points, axis=1)
        dominates[i] = False
        mask[dominates] = False
    return mask


def _plot_pareto_front(results, selection: ProblemSelection):
    plot_entries = []
    for res in results:
        F = res.get("F")
        if F is None or F.size == 0:
            continue
        algo = res.get("algorithm", "unknown").upper()
        engine = res.get("engine")
        label = f"{algo} ({engine})" if engine else algo
        plot_entries.append((label, np.asarray(F, dtype=float)))
    if not plot_entries:
        return None
    n_obj = plot_entries[0][1].shape[1]
    if n_obj < 2:
        print("Pareto visualization requires at least two objectives; skipping plot.")
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"matplotlib is required for plotting the Pareto front (skipping plot: {exc}).")
        return None

    dims = 3 if n_obj >= 3 else 2
    if dims == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_zlabel("Objective 3")
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")

    cmap = plt.cm.get_cmap("tab10", len(plot_entries))
    for idx, (label, values) in enumerate(plot_entries):
        coords = values[:, :dims]
        color = cmap(idx)
        if dims == 3:
            ax.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2], label=label, s=22, alpha=0.7, color=color
            )
        else:
            ax.scatter(coords[:, 0], coords[:, 1], label=label, s=35, alpha=0.8, color=color)

    all_points_full = np.vstack([entry[1] for entry in plot_entries])
    front_mask = _non_dominated_mask(all_points_full)
    if np.any(front_mask):
        projected = all_points_full[front_mask][:, :dims]
        if dims == 2:
            order = np.argsort(projected[:, 0])
            projected = projected[order]
            ax.plot(
                projected[:, 0],
                projected[:, 1],
                color="black",
                linewidth=2,
                label="Pareto front (union)",
            )
        else:
            ax.scatter(
                projected[:, 0],
                projected[:, 1],
                projected[:, 2],
                color="black",
                s=40,
                label="Pareto front (union)",
                marker="x",
            )

    title = f"Pareto front - {selection.spec.label}"
    if n_obj > dims:
        title += f" (showing first {dims} objectives)"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    os.makedirs(_problem_output_dir(selection), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"pareto_front_{timestamp}.png"
    plot_path = os.path.join(_problem_output_dir(selection), filename)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Pareto front plot saved to: {plot_path}")
    return plot_path


def _execute_problem_suite(args, problem_selection: ProblemSelection):
    engines = EXPERIMENT_BACKENDS if args.experiment == "backends" else (args.engine,)
    algorithms = list(ENABLED_ALGORITHMS) if args.algorithm == "both" else [args.algorithm]
    include_external = args.include_external
    use_native_external_problem = args.external_problem_source == "native"
    if include_external and problem_selection.spec.key != "zdt1":
        print(
            "External baselines are currently available only for ZDT1; "
            "skipping external runs."
        )
        include_external = False

    if include_external:
        for ext in EXTERNAL_ALGORITHM_NAMES:
            if ext not in algorithms:
                algorithms.append(ext)

    internal_algorithms = [a for a in algorithms if a in ENABLED_ALGORITHMS]
    optional_algorithms = [a for a in algorithms if a in OPTIONAL_ALGORITHMS]
    external_algorithms = [a for a in algorithms if a in EXTERNAL_ALGORITHM_NAMES]

    results = []
    for engine in engines:
        for algorithm_name in internal_algorithms:
            metrics = _run_single(
                engine, algorithm_name, problem_selection, seed=SEED
            )
            results.append(metrics)
        for algorithm_name in optional_algorithms:
            metrics = _run_single(
                engine, algorithm_name, problem_selection, seed=SEED
            )
            results.append(metrics)

    for algorithm_name in external_algorithms:
        metrics = _execute_external_runner(
            algorithm_name,
            problem_selection,
            use_native_problem=use_native_external_problem,
        )
        if metrics is not None:
            results.append(metrics)

    if not results:
        print("No runs were executed. Check algorithm selection or install missing dependencies.")
        return

    fronts = [res["F"] for res in results]
    hv_ref_point = _compute_hv_reference(fronts)
    for res in results:
        backend = res.pop("_kernel_backend", None)
        if backend and backend.supports_quality_indicator("hypervolume"):
            hv_value = backend.hypervolume(res["F"], hv_ref_point)
            res["hv_source"] = backend.__class__.__name__
        else:
            hv_value = hypervolume(res["F"], hv_ref_point)
            res["hv_source"] = "global"
        res["hv"] = hv_value

    if len(results) == 1:
        hv_val = results[0]["hv"]
        ref_txt = np.array2string(hv_ref_point, precision=3, suppress_small=True)
        print(f"\nHypervolume (reference {ref_txt}): {hv_val:.6f}")
    else:
        _print_summary(results, hv_ref_point)

    _plot_pareto_front(results, problem_selection)


def main():
    args = _parse_args()
    selections = _resolve_problem_selections(args)
    multiple = len(selections) > 1
    for idx, selection in enumerate(selections, start=1):
        if multiple:
            print("\n" + "#" * 80)
            print(f"Problem {idx}/{len(selections)}: {selection.spec.label} ({selection.spec.key})")
            print("#" * 80 + "\n")
        _execute_problem_suite(args, selection)


if __name__ == "__main__":
    main()
