from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from vamos.algorithm.config import MOEADConfig, NSGAIIConfig, NSGAIIIConfig, SMSEMOAConfig
from vamos.algorithm.hypervolume import hypervolume
from vamos.algorithm.moead import MOEAD
from vamos.algorithm.nsgaii import NSGAII
from vamos.algorithm.nsga3 import NSGAIII
from vamos.algorithm.smsemoa import SMSEMOA
from vamos.kernel.numpy_backend import NumPyKernel
from vamos.problem.registry import ProblemSelection, available_problem_names, make_problem_selection
from vamos.problem.types import ProblemProtocol, MixedProblemProtocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TITLE = "VAMOS: Vectorized Architecture for Multiobjective Optimization Studies"
DEFAULT_ALGORITHM = "nsgaii"
DEFAULT_ENGINE = "numpy"
DEFAULT_PROBLEM = "zdt1"

ENABLED_ALGORITHMS = ("nsgaii", "moead", "smsemoa")
OPTIONAL_ALGORITHMS = ("nsgaiii",)
EXTERNAL_ALGORITHM_NAMES = ("pymoo_nsga2", "jmetalpy_nsga2", "pygmo_nsga2")
HV_REFERENCE_OFFSET = 0.1

EXPERIMENT_BACKENDS = (
    "numpy",
    "numba",
    "moocore",
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

REFERENCE_FRONT_PATHS = {
    "zdt1": PROJECT_ROOT / "data/reference_fronts/ZDT1.csv",
    "zdt2": PROJECT_ROOT / "data/reference_fronts/ZDT2.csv",
    "zdt3": PROJECT_ROOT / "data/reference_fronts/ZDT3.csv",
    "zdt4": PROJECT_ROOT / "data/reference_fronts/ZDT4.csv",
    "zdt6": PROJECT_ROOT / "data/reference_fronts/ZDT6.csv",
}


@dataclass(frozen=True)
class ExperimentConfig:
    title: str = TITLE
    output_root: str = os.environ.get("VAMOS_OUTPUT_ROOT", "results")
    population_size: int = 100
    offspring_population_size: int | None = None
    max_evaluations: int = 25000
    seed: int = 42

    def offspring_size(self) -> int:
        return self.offspring_population_size or self.population_size


def resolve_reference_front_path(problem_key: str, explicit_path: str | None):
    if explicit_path:
        path = Path(explicit_path)
    else:
        candidate = REFERENCE_FRONT_PATHS.get(problem_key.lower())
        if candidate is None:
            raise FileNotFoundError(
                f"No default reference front is available for problem '{problem_key}'. "
                "Provide --hv-reference-front."
            )
        path = Path(candidate)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reference front file '{path}' does not exist.")
    return path


def build_hv_stop_config(hv_threshold: float | None, hv_reference_front: str | None, problem_key: str):
    if hv_threshold is None:
        return None
    front_path = resolve_reference_front_path(problem_key, hv_reference_front)
    reference_front = np.loadtxt(front_path, delimiter=",")
    if reference_front.ndim != 2 or reference_front.shape[1] < 2:
        raise ValueError(
            f"Reference front '{front_path}' must be a 2D array with at least two objectives."
        )
    max_vals = reference_front.max(axis=0)
    margin = np.maximum(0.1 * np.maximum(np.abs(max_vals), 1.0), 5.0)
    if problem_key.lower() == "zdt6":
        margin = np.maximum(margin, 10.0)
    ref_point = max_vals + margin
    hv_full = hypervolume(reference_front, ref_point)
    if hv_full <= 0.0:
        raise ValueError(
            f"Reference front '{front_path}' produced a non-positive hypervolume; check the data."
        )
    threshold_fraction = float(hv_threshold)
    return {
        "target_value": hv_full * threshold_fraction,
        "threshold_fraction": threshold_fraction,
        "reference_point": ref_point.tolist(),
        "reference_front_path": str(front_path),
    }


def resolve_nsgaii_variation_config(encoding: str, overrides: dict | None):
    overrides = overrides or {}
    cross_overrides = overrides.get("crossover") or {}
    mutation_overrides = overrides.get("mutation") or {}
    repair_choice = overrides.get("repair") or "clip"

    if encoding == "permutation":
        perm_crossovers = {
            "ox",
            "order",
            "oxd",
            "pmx",
            "cycle",
            "cx",
            "position",
            "position_based",
            "pos",
            "edge",
            "edge_recombination",
            "erx",
        }
        perm_mutations = {
            "swap",
            "insert",
            "scramble",
            "inversion",
            "simple_inversion",
            "simpleinv",
            "displacement",
        }
        cross_method_candidate = (cross_overrides.get("method") or "").lower()
        if cross_method_candidate and cross_method_candidate not in perm_crossovers:
            raise ValueError(f"Unsupported NSGA-II crossover '{cross_method_candidate}' for permutation encoding.")
        cross_method = cross_method_candidate or "ox"
        cross_params: dict[str, float | str] = {"prob": cross_overrides.get("prob", 0.9)}

        mutation_method_candidate = (mutation_overrides.get("method") or "").lower()
        if mutation_method_candidate and mutation_method_candidate not in perm_mutations:
            raise ValueError(f"Unsupported NSGA-II mutation '{mutation_method_candidate}' for permutation encoding.")
        mutation_method = mutation_method_candidate or "swap"
        mut_params: dict[str, float | str] = {"prob": mutation_overrides.get("prob", "2/n")}
        return (cross_method, cross_params), (mutation_method, mut_params), None

    if encoding == "binary":
        binary_crossovers = {
            "one_point",
            "single_point",
            "1point",
            "two_point",
            "2point",
            "uniform",
        }
        binary_mutations = {"bitflip", "bit_flip"}

        cross_method_candidate = (cross_overrides.get("method") or "").lower()
        if cross_method_candidate and cross_method_candidate not in binary_crossovers:
            raise ValueError(f"Unsupported NSGA-II crossover '{cross_method_candidate}' for binary encoding.")
        cross_method = cross_method_candidate or "uniform"
        cross_params: dict[str, float | str] = {"prob": cross_overrides.get("prob", 0.9)}

        mutation_method_candidate = (mutation_overrides.get("method") or "").lower()
        if mutation_method_candidate and mutation_method_candidate not in binary_mutations:
            raise ValueError(f"Unsupported NSGA-II mutation '{mutation_method_candidate}' for binary encoding.")
        mutation_method = mutation_method_candidate or "bitflip"
        mut_params: dict[str, float | str] = {"prob": mutation_overrides.get("prob", "1/n")}
        return (cross_method, cross_params), (mutation_method, mut_params), None

    if encoding == "integer":
        int_crossovers = {"uniform", "blend", "arithmetic"}
        int_mutations = {"reset", "random_reset", "creep"}

        cross_method_candidate = (cross_overrides.get("method") or "").lower()
        if cross_method_candidate and cross_method_candidate not in int_crossovers:
            raise ValueError(f"Unsupported NSGA-II crossover '{cross_method_candidate}' for integer encoding.")
        cross_method = cross_method_candidate or "uniform"
        cross_params: dict[str, float | str] = {"prob": cross_overrides.get("prob", 0.9)}

        mutation_method_candidate = (mutation_overrides.get("method") or "").lower()
        if mutation_method_candidate and mutation_method_candidate not in int_mutations:
            raise ValueError(f"Unsupported NSGA-II mutation '{mutation_method_candidate}' for integer encoding.")
        mutation_method = mutation_method_candidate or "reset"
        mut_params: dict[str, float | str] = {
            "prob": mutation_overrides.get("prob", "1/n"),
            "step": mutation_overrides.get("step", 1),
        }
        return (cross_method, cross_params), (mutation_method, mut_params), None

    if encoding == "mixed":
        mixed_crossovers = {"mixed", "uniform"}
        mixed_mutations = {"mixed", "gaussian"}

        cross_method_candidate = (cross_overrides.get("method") or "").lower()
        if cross_method_candidate and cross_method_candidate not in mixed_crossovers:
            raise ValueError(f"Unsupported NSGA-II crossover '{cross_method_candidate}' for mixed encoding.")
        cross_method = cross_method_candidate or "mixed"
        cross_params: dict[str, float | str] = {"prob": cross_overrides.get("prob", 0.9)}

        mutation_method_candidate = (mutation_overrides.get("method") or "").lower()
        if mutation_method_candidate and mutation_method_candidate not in mixed_mutations:
            raise ValueError(f"Unsupported NSGA-II mutation '{mutation_method_candidate}' for mixed encoding.")
        mutation_method = mutation_method_candidate or "mixed"
        mut_params: dict[str, float | str] = {"prob": mutation_overrides.get("prob", "1/n")}
        return (cross_method, cross_params), (mutation_method, mut_params), None

    cross_method = (cross_overrides.get("method") or "sbx").lower()
    if cross_method not in {"sbx", "blx_alpha"}:
        raise ValueError(f"Unsupported NSGA-II crossover '{cross_method}'.")
    cross_params: dict[str, float | str] = {"prob": 0.9}
    cross_prob_override = cross_overrides.get("prob")
    if cross_prob_override is not None:
        cross_params["prob"] = cross_prob_override
    if cross_method == "sbx":
        eta_override = cross_overrides.get("eta")
        cross_params["eta"] = eta_override if eta_override is not None else 20.0
    else:
        alpha_override = cross_overrides.get("alpha")
        cross_params["alpha"] = alpha_override if alpha_override is not None else 0.5

    mutation_method = (mutation_overrides.get("method") or "pm").lower()
    if mutation_method not in {"pm", "non_uniform"}:
        raise ValueError(f"Unsupported NSGA-II mutation '{mutation_method}'.")
    mut_params: dict[str, float | str] = {"prob": "1/n"}
    mut_prob_override = mutation_overrides.get("prob")
    if mut_prob_override is not None:
        mut_params["prob"] = mut_prob_override
    if mutation_method == "pm":
        eta_override = mutation_overrides.get("eta")
        mut_params["eta"] = eta_override if eta_override is not None else 20.0
    else:
        pert_override = mutation_overrides.get("perturbation")
        mut_params["perturbation"] = pert_override if pert_override is not None else 0.5

    repair_cfg = None
    if repair_choice and repair_choice.lower() != "none":
        repair_cfg = (repair_choice.lower(), {})

    return (cross_method, cross_params), (mutation_method, mut_params), repair_cfg


def serialize_operator_tuple(op_tuple):
    if not op_tuple:
        return None
    name, params = op_tuple
    return {"name": name, "params": params}


def collect_operator_metadata(cfg_data) -> dict:
    if cfg_data is None:
        return {}
    payload = {}
    for key in ("crossover", "mutation", "repair"):
        value = getattr(cfg_data, key, None)
        formatted = serialize_operator_tuple(value)
        if formatted:
            payload[key] = formatted
    return payload


def resolve_kernel(engine_name: str):
    if engine_name == "numpy":
        return NumPyKernel()
    if engine_name == "numba":
        try:
            from vamos.kernel.numba_backend import NumbaKernel
        except ImportError as exc:
            raise SystemExit(
                "The 'numba' backend requires the numba dependency to be installed.\n"
                f"Original error: {exc}"
            ) from exc
        return NumbaKernel()
    if engine_name == "moocore":
        try:
            from vamos.kernel.moocore_backend import MooCoreKernel
        except ImportError as exc:
            raise SystemExit(
                "The 'moocore' backend requires the moocore dependency to be installed.\n"
                f"Original error: {exc}"
            ) from exc
        return MooCoreKernel()
    raise ValueError(f"Unsupported kernel backend '{engine_name}'.")


def compute_hv_reference(fronts: list[np.ndarray]) -> np.ndarray:
    stacked = np.vstack(fronts)
    return stacked.max(axis=0) + HV_REFERENCE_OFFSET


def _merge_variation_overrides(base: dict | None, override: dict | None) -> dict:
    merged = deepcopy(base) if base else {}
    if not override:
        return merged
    for key in ("crossover", "mutation"):
        if key in override and isinstance(override[key], dict):
            merged.setdefault(key, {})
            for k, v in override[key].items():
                if v is not None:
                    merged[key][k] = v
    if "repair" in override:
        merged["repair"] = override["repair"]
    return merged


def _validate_problem(problem: ProblemProtocol) -> None:
    if problem.n_var <= 0 or problem.n_obj <= 0:
        raise ValueError("Problem must have positive n_var and n_obj.")
    xl = np.asarray(problem.xl)
    xu = np.asarray(problem.xu)
    if xl.ndim > 1 or xu.ndim > 1:
        raise ValueError("Problem bounds must be scalars or 1D arrays.")
    if xl.ndim == 1 and xl.shape[0] != problem.n_var:
        raise ValueError("Lower bounds length must match n_var.")
    if xu.ndim == 1 and xu.shape[0] != problem.n_var:
        raise ValueError("Upper bounds length must match n_var.")
    if np.any(xl > xu):
        raise ValueError("Lower bounds must not exceed upper bounds.")
    encoding = getattr(problem, "encoding", "continuous")
    if encoding == "mixed":
        if not hasattr(problem, "mixed_spec"):
            raise ValueError("Mixed-encoding problems must define 'mixed_spec'.")
        spec = getattr(problem, "mixed_spec")
        required = {"real_idx", "int_idx", "cat_idx", "real_lower", "real_upper", "int_lower", "int_upper", "cat_cardinality"}
        missing = required - set(spec.keys())
        if missing:
            raise ValueError(f"mixed_spec missing required fields: {', '.join(sorted(missing))}")


def problem_output_dir(selection: ProblemSelection, config: ExperimentConfig) -> str:
    safe = selection.spec.label.replace(" ", "_").upper()
    return os.path.join(config.output_root, f"{safe}")


def run_output_dir(
    selection: ProblemSelection, algorithm_name: str, engine_name: str, seed: int, config: ExperimentConfig
) -> str:
    base = problem_output_dir(selection, config)
    return os.path.join(
        base,
        algorithm_name.lower(),
        engine_name.lower(),
        f"seed_{seed}",
    )


def resolve_problem_selection(args) -> ProblemSelection:
    try:
        return make_problem_selection(
            args.problem, n_var=args.n_var, n_obj=args.n_obj
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def resolve_problem_selections(args) -> list[ProblemSelection]:
    if not getattr(args, "problem_set", None):
        return [resolve_problem_selection(args)]
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


def _default_weight_path(problem_name: str, n_obj: int, pop_size: int) -> str:
    filename = f"{problem_name}_nobj{n_obj}_pop{pop_size}.csv"
    return str(PROJECT_ROOT / "build" / "weights" / filename)


def _build_algorithm(
    algorithm_name: str,
    engine_name: str,
    problem,
    config: ExperimentConfig,
    *,
    external_archive_size: int | None = None,
    selection_pressure: int = 2,
    nsgaii_variation: dict | None = None,
    moead_variation: dict | None = None,
    smsemoa_variation: dict | None = None,
    nsga3_variation: dict | None = None,
):
    kernel_backend = resolve_kernel(engine_name)
    encoding = getattr(problem, "encoding", "continuous")
    if encoding in {"permutation", "mixed"} and algorithm_name != "nsgaii":
        raise ValueError(
            f"Problem '{problem.__class__.__name__}' uses {encoding} encoding; "
            f"currently only NSGA-II supports this representation."
        )
    if algorithm_name == "nsgaii":
        crossover_cfg, mutation_cfg, repair_cfg = resolve_nsgaii_variation_config(
            encoding, nsgaii_variation
        )
        cfg_builder = (
            NSGAIIConfig()
            .pop_size(config.population_size)
            .crossover(crossover_cfg[0], **crossover_cfg[1])
            .mutation(mutation_cfg[0], **mutation_cfg[1])
            .selection("tournament", pressure=selection_pressure)
            .survival("nsga2")
            .engine(engine_name)
            .offspring_size(config.offspring_size())
        )
        if repair_cfg:
            cfg_builder = cfg_builder.repair(repair_cfg[0], **repair_cfg[1])
        if external_archive_size:
            cfg_builder = cfg_builder.external_archive(size=external_archive_size)
        cfg = cfg_builder.fixed()
        return NSGAII(cfg.to_dict(), kernel=kernel_backend), cfg

    if algorithm_name == "moead":
        weight_path = _default_weight_path(
            problem.__class__.__name__, problem.n_obj, config.population_size
        )
        mv = moead_variation or {}
        if encoding == "binary":
            cross_cfg = (mv.get("crossover", {}).get("method") or "uniform", {"prob": mv.get("crossover", {}).get("prob", 0.9)})
            mut_cfg = (mv.get("mutation", {}).get("method") or "bitflip", {"prob": mv.get("mutation", {}).get("prob", "1/n")})
        elif encoding == "integer":
            cross_cfg = (mv.get("crossover", {}).get("method") or "uniform", {"prob": mv.get("crossover", {}).get("prob", 0.9)})
            mut_cfg = (
                mv.get("mutation", {}).get("method") or "reset",
                {
                    "prob": mv.get("mutation", {}).get("prob", "1/n"),
                    "step": mv.get("mutation", {}).get("step", 1),
                },
            )
        elif encoding in {"continuous", "real"}:
            cross_cfg = (
                mv.get("crossover", {}).get("method") or "sbx",
                {
                    "prob": mv.get("crossover", {}).get("prob", 0.9),
                    "eta": mv.get("crossover", {}).get("eta", 20.0),
                },
            )
            mut_cfg = (
                mv.get("mutation", {}).get("method") or "pm",
                {
                    "prob": mv.get("mutation", {}).get("prob", "1/n"),
                    "eta": mv.get("mutation", {}).get("eta", 20.0),
                },
            )
        else:
            raise ValueError(
                f"Problem '{problem.__class__.__name__}' uses unsupported encoding '{encoding}' for MOEA/D."
            )
        cfg_builder = (
            MOEADConfig()
            .pop_size(config.population_size)
            .neighbor_size(min(20, config.population_size))
            .delta(0.9)
            .replace_limit(2)
            .crossover(cross_cfg[0], **cross_cfg[1])
            .mutation(mut_cfg[0], **mut_cfg[1])
            .aggregation("tchebycheff")
            .weight_vectors(path=weight_path)
            .engine(engine_name)
        )
        cfg = cfg_builder.fixed()
        return MOEAD(cfg.to_dict(), kernel=kernel_backend), cfg

    if algorithm_name == "smsemoa":
        sv = smsemoa_variation or {}
        if encoding == "binary":
            cross_cfg = (sv.get("crossover", {}).get("method") or "uniform", {"prob": sv.get("crossover", {}).get("prob", 0.9)})
            mut_cfg = (sv.get("mutation", {}).get("method") or "bitflip", {"prob": sv.get("mutation", {}).get("prob", "1/n")})
        elif encoding == "integer":
            cross_cfg = (sv.get("crossover", {}).get("method") or "uniform", {"prob": sv.get("crossover", {}).get("prob", 0.9)})
            mut_cfg = (
                sv.get("mutation", {}).get("method") or "reset",
                {
                    "prob": sv.get("mutation", {}).get("prob", "1/n"),
                    "step": sv.get("mutation", {}).get("step", 1),
                },
            )
        elif encoding in {"continuous", "real"}:
            cross_cfg = (
                sv.get("crossover", {}).get("method") or "sbx",
                {
                    "prob": sv.get("crossover", {}).get("prob", 0.9),
                    "eta": sv.get("crossover", {}).get("eta", 20.0),
                },
            )
            mut_cfg = (
                sv.get("mutation", {}).get("method") or "pm",
                {
                    "prob": sv.get("mutation", {}).get("prob", "1/n"),
                    "eta": sv.get("mutation", {}).get("eta", 20.0),
                },
            )
        else:
            raise ValueError(
                f"Problem '{problem.__class__.__name__}' uses unsupported encoding '{encoding}' for SMSEMOA."
            )
        cfg_builder = (
            SMSEMOAConfig()
            .pop_size(config.population_size)
            .crossover(cross_cfg[0], **cross_cfg[1])
            .mutation(mut_cfg[0], **mut_cfg[1])
            .selection("tournament", pressure=selection_pressure)
            .reference_point(offset=0.1, adaptive=True)
            .engine(engine_name)
        )
        cfg = cfg_builder.fixed()
        return SMSEMOA(cfg.to_dict(), kernel=kernel_backend), cfg

    if algorithm_name == "nsgaiii":
        ref_path = _default_weight_path(
            problem.__class__.__name__, problem.n_obj, config.population_size
        )
        nv = nsga3_variation or {}
        if encoding == "binary":
            cross_cfg = (nv.get("crossover", {}).get("method") or "uniform", {"prob": nv.get("crossover", {}).get("prob", 0.9)})
            mut_cfg = (nv.get("mutation", {}).get("method") or "bitflip", {"prob": nv.get("mutation", {}).get("prob", "1/n")})
        elif encoding == "integer":
            cross_cfg = (nv.get("crossover", {}).get("method") or "uniform", {"prob": nv.get("crossover", {}).get("prob", 0.9)})
            mut_cfg = (
                nv.get("mutation", {}).get("method") or "reset",
                {
                    "prob": nv.get("mutation", {}).get("prob", "1/n"),
                    "step": nv.get("mutation", {}).get("step", 1),
                },
            )
        elif encoding in {"continuous", "real"}:
            cross_cfg = (
                nv.get("crossover", {}).get("method") or "sbx",
                {
                    "prob": nv.get("crossover", {}).get("prob", 0.9),
                    "eta": nv.get("crossover", {}).get("eta", 20.0),
                },
            )
            mut_cfg = (
                nv.get("mutation", {}).get("method") or "pm",
                {
                    "prob": nv.get("mutation", {}).get("prob", "1/n"),
                    "eta": nv.get("mutation", {}).get("eta", 20.0),
                },
            )
        else:
            raise ValueError(
                f"Problem '{problem.__class__.__name__}' uses unsupported encoding '{encoding}' for NSGA-III."
            )
        cfg_builder = (
            NSGAIIIConfig()
            .pop_size(config.population_size)
            .crossover(cross_cfg[0], **cross_cfg[1])
            .mutation(mut_cfg[0], **mut_cfg[1])
            .selection("tournament", pressure=selection_pressure)
            .reference_directions(path=ref_path)
            .engine(engine_name)
        )
        cfg = cfg_builder.fixed()
        return NSGAIII(cfg.to_dict(), kernel=kernel_backend), cfg

    raise ValueError(f"Unsupported algorithm '{algorithm_name}'.")


def _print_run_banner(
    problem, problem_selection: ProblemSelection, algorithm_label: str, backend_label: str, config: ExperimentConfig
):
    print("=" * 80)
    print(config.title)
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
    print(f"Population size: {config.population_size}")
    print(f"Offspring population size: {config.offspring_size()}")
    print(f"Max evaluations: {config.max_evaluations}")
    print("-" * 80)


def _make_metrics(
    algorithm_name: str,
    engine_name: str,
    total_time_ms: float,
    evaluations: int,
    F: np.ndarray,
):
    spread = None
    if F.size and F.shape[1] >= 1:
        spread = np.ptp(F[:, 0])
    evals_per_sec = evaluations / max(1e-9, total_time_ms / 1000.0)
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
    algo = metrics["algorithm"]
    time_ms = metrics["time_ms"]
    evals = metrics["evaluations"]
    hv_info = ""
    hv = metrics.get("hv")
    if hv is not None:
        hv_info = f" | HV: {hv:.6f}"
    print(f"{algo} -> Time: {time_ms:.2f} ms | Eval/s: {metrics['evals_per_sec']:.1f}{hv_info}")
    spread = metrics.get("spread")
    if spread is not None:
        print(f"Objective 1 spread: {spread:.6f}")


def _build_run_metadata(
    selection: ProblemSelection,
    algorithm_name: str,
    engine_name: str,
    cfg_data,
    metrics: dict,
    *,
    kernel_backend,
    seed: int,
    config: ExperimentConfig,
):
    timestamp = datetime.utcnow().isoformat()
    problem = selection.instantiate()
    problem_info = {
        "label": selection.spec.label,
        "key": selection.spec.key,
        "n_var": selection.n_var,
        "n_obj": selection.n_obj,
        "encoding": getattr(problem, "encoding", "continuous"),
    }
    try:
        problem_info["description"] = selection.spec.description
    except Exception:
        pass

    kernel_caps = sorted(set(kernel_backend.capabilities())) if kernel_backend else []
    kernel_info = {
        "name": kernel_backend.__class__.__name__ if kernel_backend else "external",
        "device": kernel_backend.device() if kernel_backend else "external",
        "capabilities": kernel_caps,
    }
    operator_payload = collect_operator_metadata(cfg_data)
    config_payload = cfg_data.to_dict() if hasattr(cfg_data, "to_dict") else None
    metric_payload = {
        "time_ms": metrics["time_ms"],
        "evaluations": metrics["evaluations"],
        "evals_per_sec": metrics["evals_per_sec"],
        "spread": metrics["spread"],
        "termination": metrics.get("termination"),
    }
    if metrics.get("hv_threshold_fraction") is not None:
        metric_payload["hv_threshold_fraction"] = metrics.get("hv_threshold_fraction")
        metric_payload["hv_reference_point"] = metrics.get("hv_reference_point")
        metric_payload["hv_reference_front"] = metrics.get("hv_reference_front")
    metadata = {
        "title": config.title,
        "timestamp": timestamp,
        "algorithm": algorithm_name,
        "backend": engine_name,
        "backend_info": kernel_info,
        "seed": seed,
        "population_size": config.population_size,
        "max_evaluations": config.max_evaluations,
        "problem": problem_info,
        "config": config_payload,
        "metrics": metric_payload,
    }
    if operator_payload:
        metadata["operators"] = operator_payload
    return metadata


def run_single(
    engine_name: str,
    algorithm_name: str,
    selection: ProblemSelection,
    config: ExperimentConfig,
    *,
    external_archive_size: int | None = None,
    selection_pressure: int = 2,
    nsgaii_variation: dict | None = None,
    moead_variation: dict | None = None,
    smsemoa_variation: dict | None = None,
    nsga3_variation: dict | None = None,
    hv_stop_config: dict | None = None,
    config_source: str | None = None,
    problem_override: dict | None = None,
):
    problem = selection.instantiate()
    display_algo = algorithm_name.upper()
    _print_run_banner(problem, selection, display_algo, engine_name, config)
    algorithm, cfg_data = _build_algorithm(
        algorithm_name,
        engine_name,
        problem,
        config,
        external_archive_size=external_archive_size,
        selection_pressure=selection_pressure,
        nsgaii_variation=nsgaii_variation,
        moead_variation=moead_variation,
        smsemoa_variation=smsemoa_variation,
        nsga3_variation=nsga3_variation,
    )
    kernel_backend = algorithm.kernel

    hv_termination = None
    termination = ("n_eval", config.max_evaluations)
    hv_enabled = hv_stop_config is not None and algorithm_name == "nsgaii"
    if hv_enabled:
        hv_termination = dict(hv_stop_config)
        hv_termination["max_evaluations"] = config.max_evaluations
        termination = ("hv", hv_termination)

    _validate_problem(problem)

    start = time.perf_counter()
    result = algorithm.run(problem, termination=termination, seed=config.seed)
    end = time.perf_counter()

    total_time_ms = (end - start) * 1000.0
    F = result["F"]
    archive = result.get("archive")
    actual_evaluations = int(result.get("evaluations", config.max_evaluations))
    termination_reason = "max_evaluations"
    if hv_enabled and result.get("hv_reached"):
        termination_reason = "hv_threshold"

    metrics = _make_metrics(
        algorithm_name, engine_name, total_time_ms, actual_evaluations, F
    )
    metrics["termination"] = termination_reason
    if hv_enabled and hv_stop_config:
        metrics["hv_threshold_fraction"] = hv_stop_config.get("threshold_fraction")
        metrics["hv_reference_point"] = hv_stop_config.get("reference_point")
        metrics["hv_reference_front"] = hv_stop_config.get("reference_front_path")
    metrics["config"] = cfg_data
    if kernel_backend is not None:
        metrics["_kernel_backend"] = kernel_backend
        metrics["backend_device"] = kernel_backend.device()
        metrics["backend_capabilities"] = sorted(set(kernel_backend.capabilities()))
    else:
        metrics["backend_device"] = "external"
        metrics["backend_capabilities"] = []
    _print_run_results(metrics)
    output_dir = run_output_dir(selection, algorithm_name, engine_name, config.seed, config)
    metrics["output_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)
    fun_path = os.path.join(output_dir, "FUN.csv")
    np.savetxt(fun_path, F, delimiter=",")
    archive_fun_path = None
    if archive is not None:
        archive_fun_path = os.path.join(output_dir, "ARCHIVE_FUN.csv")
        np.savetxt(archive_fun_path, archive["F"], delimiter=",")
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
        seed=config.seed,
        config=config,
    )
    metadata["config_source"] = config_source
    if problem_override:
        metadata["problem_override"] = problem_override
    if hv_stop_config:
        metadata["hv_stop_config"] = hv_stop_config
    metadata["artifacts"] = {"fun": "FUN.csv", "time_ms": "time.txt"}
    if archive_fun_path:
        metadata["artifacts"]["archive_fun"] = os.path.basename(archive_fun_path)
    resolved_cfg = {
        "algorithm": algorithm_name,
        "engine": engine_name,
        "problem": selection.spec.key,
        "n_var": selection.n_var,
        "n_obj": selection.n_obj,
        "encoding": getattr(problem, "encoding", "continuous"),
        "population_size": config.population_size,
        "offspring_population_size": config.offspring_size(),
        "max_evaluations": config.max_evaluations,
        "seed": config.seed,
        "selection_pressure": selection_pressure,
        "external_archive_size": external_archive_size,
        "hv_threshold": hv_stop_config.get("threshold_fraction") if hv_stop_config else None,
        "hv_reference_point": hv_stop_config.get("reference_point") if hv_stop_config else None,
        "hv_reference_front": hv_stop_config.get("reference_front_path") if hv_stop_config else None,
        "nsgaii_variation": nsgaii_variation,
        "moead_variation": moead_variation,
        "smsemoa_variation": smsemoa_variation,
        "nsga3_variation": nsga3_variation,
        "config_source": config_source,
        "problem_override": problem_override,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "resolved_config.json"), "w", encoding="utf-8") as f:
        json.dump(resolved_cfg, f, indent=2, sort_keys=True)

    print("\nResults stored in:", output_dir)
    print("=" * 80)

    return metrics


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
        hv_txt = f"{res['hv']:.6f}" if res.get("hv") is not None else "-"
        print(
            f"{res['algorithm']:<12} {res['engine']:<10} {res['time_ms']:>12.2f} "
            f"{res['evals_per_sec']:>12.1f} {hv_txt:>12} {spread_txt:>12}"
        )
    ref_txt = np.array2string(hv_ref_point, precision=3, suppress_small=True)
    print(f"\nHypervolume reference point: {ref_txt}")


def execute_problem_suite(
    args,
    problem_selection: ProblemSelection,
    config: ExperimentConfig,
    *,
    hv_stop_config: dict | None = None,
    nsgaii_variation: dict | None = None,
    include_external: bool = False,
    config_source: str | None = None,
    problem_override: dict | None = None,
):
    from vamos import external  # local import to keep runner decoupled
    from vamos import plotting

    engines: Iterable[str] = EXPERIMENT_BACKENDS if args.experiment == "backends" else (args.engine,)
    algorithms = list(ENABLED_ALGORITHMS) if args.algorithm == "both" else [args.algorithm]
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
            metrics = run_single(
                engine,
                algorithm_name,
                problem_selection,
                config,
                external_archive_size=args.external_archive_size,
                selection_pressure=args.selection_pressure,
                nsgaii_variation=nsgaii_variation,
                moead_variation=getattr(args, "moead_variation", None),
                smsemoa_variation=getattr(args, "smsemoa_variation", None),
                nsga3_variation=getattr(args, "nsga3_variation", None),
                hv_stop_config=hv_stop_config if algorithm_name == "nsgaii" else None,
                config_source=config_source,
                problem_override=problem_override,
            )
            results.append(metrics)
        for algorithm_name in optional_algorithms:
            metrics = run_single(
                engine,
                algorithm_name,
                problem_selection,
                config,
                external_archive_size=args.external_archive_size,
                selection_pressure=args.selection_pressure,
                nsgaii_variation=nsgaii_variation,
                moead_variation=getattr(args, "moead_variation", None),
                smsemoa_variation=getattr(args, "smsemoa_variation", None),
                nsga3_variation=getattr(args, "nsga3_variation", None),
                hv_stop_config=hv_stop_config if algorithm_name == "nsgaii" else None,
                config_source=config_source,
                problem_override=problem_override,
            )
            results.append(metrics)

    for algorithm_name in external_algorithms:
        metrics = external.run_external(
            algorithm_name,
            problem_selection,
            use_native_problem=use_native_external_problem,
            config=config,
            make_metrics=_make_metrics,
            print_banner=lambda problem, selection, label, backend: _print_run_banner(
                problem, selection, label, backend, config
            ),
            print_results=_print_run_results,
        )
        if metrics is not None:
            results.append(metrics)

    if not results:
        print("No runs were executed. Check algorithm selection or install missing dependencies.")
        return

    fronts = [res["F"] for res in results]
    hv_ref_point = compute_hv_reference(fronts)
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

    plotting.plot_pareto_front(results, problem_selection, output_root=config.output_root, title=config.title)


def run_from_args(args, config: ExperimentConfig):
    selections = resolve_problem_selections(args)
    multiple = len(selections) > 1
    base_variation = getattr(args, "nsgaii_variation", None)
    overrides = getattr(args, "problem_overrides", {}) or {}
    config_source = getattr(args, "config_path", None)

    for idx, selection in enumerate(selections, start=1):
        override = overrides.get(selection.spec.key, {}) or {}
        effective_selection = selection
        if override.get("n_var") is not None or override.get("n_obj") is not None:
            effective_selection = make_problem_selection(
                selection.spec.key,
                n_var=override.get("n_var", selection.n_var),
                n_obj=override.get("n_obj", selection.n_obj),
            )
        effective_config = ExperimentConfig(
            title=override.get("title", config.title),
            output_root=override.get("output_root", config.output_root),
            population_size=override.get("population_size", config.population_size),
            offspring_population_size=override.get(
                "offspring_population_size", config.offspring_population_size
            ),
            max_evaluations=override.get("max_evaluations", config.max_evaluations),
            seed=override.get("seed", config.seed),
        )
        effective_args = deepcopy(args)
        for key in ("algorithm", "engine", "experiment", "include_external", "external_problem_source"):
            if override.get(key) is not None:
                setattr(effective_args, key, override[key])
        effective_args.selection_pressure = override.get("selection_pressure", args.selection_pressure)
        effective_args.external_archive_size = override.get("external_archive_size", args.external_archive_size)
        effective_args.hv_threshold = override.get("hv_threshold", args.hv_threshold)
        effective_args.hv_reference_front = override.get("hv_reference_front", args.hv_reference_front)
        effective_args.n_var = override.get("n_var", args.n_var)
        effective_args.n_obj = override.get("n_obj", args.n_obj)
        effective_args.nsgaii_variation = _merge_variation_overrides(base_variation, override.get("nsgaii"))
        effective_args.moead_variation = _merge_variation_overrides(getattr(args, "moead_variation", None), override.get("moead"))
        effective_args.smsemoa_variation = _merge_variation_overrides(getattr(args, "smsemoa_variation", None), override.get("smsemoa"))
        effective_args.nsga3_variation = _merge_variation_overrides(getattr(args, "nsga3_variation", None), override.get("nsga3"))
        effective_args.effective_problem_override = override

        if multiple:
            print("\n" + "#" * 80)
            print(
                f"Problem {idx}/{len(selections)}: {effective_selection.spec.label} "
                f"({effective_selection.spec.key})"
            )
            print("#" * 80 + "\n")

        hv_stop_config = None
        if effective_args.hv_threshold is not None:
            hv_stop_config = build_hv_stop_config(
                effective_args.hv_threshold, effective_args.hv_reference_front, effective_selection.spec.key
            )
        nsgaii_variation = getattr(effective_args, "nsgaii_variation", None)
        execute_problem_suite(
            effective_args,
            effective_selection,
            effective_config,
            hv_stop_config=hv_stop_config,
            nsgaii_variation=nsgaii_variation,
            include_external=effective_args.include_external,
            config_source=config_source,
            problem_override=override,
        )
