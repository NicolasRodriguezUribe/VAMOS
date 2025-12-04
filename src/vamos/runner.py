from __future__ import annotations

import os
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from vamos import plotting
from vamos.algorithm.hypervolume import hypervolume
from vamos.execution import execute_algorithm
from vamos.problem.registry import make_problem_selection
from vamos.problem.types import ProblemProtocol, MixedProblemProtocol
from vamos.io_utils import write_population, write_metadata, write_timing, ensure_dir
from vamos.version import __version__

# New imports
from vamos.experiment_config import (
    ExperimentConfig,
    TITLE,
    DEFAULT_ALGORITHM,
    DEFAULT_ENGINE,
    DEFAULT_PROBLEM,
    ENABLED_ALGORITHMS,
    OPTIONAL_ALGORITHMS,
    EXTERNAL_ALGORITHM_NAMES,
    EXPERIMENT_BACKENDS,
)
from vamos.problem.resolver import resolve_problem_selections, ProblemSelection
from vamos.algorithm.factory import build_algorithm, _merge_variation_overrides
from vamos.kernel.registry import resolve_kernel
from vamos.hv_stop import build_hv_stop_config, compute_hv_reference

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git_revision() -> Optional[str]:
    """
    Return current git commit hash if available, otherwise None.
    Safe to call in packaged installations without git.
    """
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return rev.decode().strip() or None


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





def _default_weight_path(problem_name: str, n_obj: int, pop_size: int) -> str:
    filename = f"{problem_name}_nobj{n_obj}_pop{pop_size}.csv"
    return str(PROJECT_ROOT / "build" / "weights" / filename)





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
        "vamos_version": __version__,
        "git_revision": _git_revision(),
        "problem": problem_info,
        "config": config_payload,
        "metrics": metric_payload,
    }
    if operator_payload:
        metadata["operators"] = operator_payload
    return metadata


def _normalize_operator_tuple(spec) -> tuple[str, dict] | None:
    """
    Accepts operator specs coming from CLI/config (tuple, dict, or string) and
    normalizes them to (name, params) tuples expected by the factory.
    """
    if spec is None:
        return None
    if isinstance(spec, tuple):
        return spec
    if isinstance(spec, str):
        return (spec, {})
    if isinstance(spec, dict):
        method = spec.get("method") or spec.get("name")
        if not method:
            return None
        params = {k: v for k, v in spec.items() if k not in {"method", "name"} and v is not None}
        return (method, params)
    return None


def _normalize_variation_config(raw: dict | None) -> dict | None:
    """
    Normalize variation configuration received from CLI/config-file parsing into
    the tuples expected by build_algorithm().
    """
    if not raw:
        return None
    normalized: dict = {}
    known_op_keys = {"crossover", "mutation", "selection", "repair", "aggregation"}
    for key in known_op_keys:
        op = _normalize_operator_tuple(raw.get(key))
        if op:
            normalized[key] = op
    # Preserve any additional keys (e.g., algorithm-specific knobs) that are not None.
    for key, value in raw.items():
        if key in normalized or key in known_op_keys:
            continue
        if value is not None:
            normalized[key] = value
    return normalized or None


def run_single(
    engine_name: str,
    algorithm_name: str,
    selection: ProblemSelection,
    config: ExperimentConfig,
    *,
    external_archive_size: int | None = None,
    archive_type: str = "hypervolume",
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
    nsgaii_variation = _normalize_variation_config(nsgaii_variation)
    moead_variation = _normalize_variation_config(moead_variation)
    smsemoa_variation = _normalize_variation_config(smsemoa_variation)
    nsga3_variation = _normalize_variation_config(nsga3_variation)
    algorithm, cfg_data = build_algorithm(
        algorithm_name,
        engine_name,
        problem,
        config,
        external_archive_size=external_archive_size,
        archive_type=archive_type,
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

    exec_result = execute_algorithm(algorithm, problem, termination=termination, seed=config.seed)
    payload = exec_result.payload
    total_time_ms = exec_result.elapsed_ms
    F = payload["F"]
    archive = payload.get("archive")
    actual_evaluations = int(payload.get("evaluations", config.max_evaluations))
    termination_reason = "max_evaluations"
    if hv_enabled and payload.get("hv_reached"):
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
    ensure_dir(output_dir)
    artifacts = write_population(output_dir, F, archive)
    if archive is not None:
        metrics["archive"] = archive
    write_timing(output_dir, total_time_ms)
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
    metadata["artifacts"] = {"fun": artifacts.get("fun"), "time_ms": "time.txt"}
    if "archive_fun" in artifacts:
        metadata["artifacts"]["archive_fun"] = artifacts["archive_fun"]
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
    write_metadata(output_dir, metadata, resolved_cfg)

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
