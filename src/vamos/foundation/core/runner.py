from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import numpy as np

from importlib.resources import as_file

from vamos.ux.visualization import plotting
from vamos.engine.algorithm.components.hypervolume import hypervolume
from vamos.foundation.core.execution import execute_algorithm
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.core.io_utils import ensure_dir
from vamos.foundation.core.experiment_config import (
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
from vamos.foundation.problem.resolver import resolve_problem_selections, ProblemSelection
from vamos.engine.algorithm.factory import build_algorithm
from vamos.engine.config.variation import merge_variation_overrides, normalize_variation_config
from vamos.foundation.core.hv_stop import build_hv_stop_config, compute_hv_reference
from vamos.foundation.eval.backends import resolve_eval_backend
from vamos.foundation.data import weight_path
from vamos.foundation.core.runner_output import (
    build_metrics,
    persist_run_outputs,
    print_run_banner,
    print_run_results,
)
from vamos.foundation.core.runner_utils import validate_problem, problem_output_dir, run_output_dir

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _default_weight_path(problem_name: str, n_obj: int, pop_size: int) -> str:
    filename = f"{problem_name}_nobj{n_obj}_pop{pop_size}.csv"
    try:
        with as_file(weight_path(filename)) as p:
            return str(p)
    except Exception:
        return str(weight_path("zdt1problem_2obj_pop100.csv")) if "zdt1" in filename else str(weight_path(filename))



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
    spea2_variation: dict | None = None,
    ibea_variation: dict | None = None,
    smpso_variation: dict | None = None,
    hv_stop_config: dict | None = None,
    config_source: str | None = None,
    problem_override: dict | None = None,
    track_genealogy: bool = False,
    autodiff_constraints: bool = False,
):
    problem = selection.instantiate()
    display_algo = algorithm_name.upper()
    print_run_banner(problem, selection, display_algo, engine_name, config)
    nsgaii_variation = normalize_variation_config(nsgaii_variation)
    moead_variation = normalize_variation_config(moead_variation)
    smsemoa_variation = normalize_variation_config(smsemoa_variation)
    nsga3_variation = normalize_variation_config(nsga3_variation)
    spea2_variation = normalize_variation_config(spea2_variation)
    ibea_variation = normalize_variation_config(ibea_variation)
    smpso_variation = normalize_variation_config(smpso_variation)
    autodiff_info = None
    if autodiff_constraints:
        autodiff_info = {"status": "unavailable"}
        try:
            from vamos.foundation.constraints.autodiff import build_jax_constraint_functions
            cm = getattr(problem, "constraint_model", None)
            if callable(cm):
                cm = cm()
            if cm is not None:
                fun, jac = build_jax_constraint_functions(cm)
                autodiff_info = {
                    "status": "ok",
                    "n_constraints": len(getattr(cm, "constraints", []) or []),
                }
            else:
                autodiff_info = {"status": "no_constraint_model"}
        except Exception as exc:
            autodiff_info = {"status": "error", "message": str(exc)}
    eval_backend = resolve_eval_backend(getattr(config, "eval_backend", "serial"), n_workers=getattr(config, "n_workers", None))
    live_viz = None
    output_dir = run_output_dir(selection, algorithm_name, engine_name, config.seed, config)
    if getattr(config, "live_viz", False):
        from vamos.ux.visualization.live_viz import LiveParetoPlot

        live_viz = LiveParetoPlot(
            update_interval=getattr(config, "live_viz_interval", 5),
            max_points=getattr(config, "live_viz_max_points", 1000),
            save_final_path=os.path.join(output_dir, "live_pareto.png"),
            title=f"{selection.spec.label} (live)",
        )
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
        spea2_variation=spea2_variation,
        ibea_variation=ibea_variation,
        smpso_variation=smpso_variation,
        track_genealogy=track_genealogy,
    )
    kernel_backend = algorithm.kernel

    hv_termination = None
    termination = ("n_eval", config.max_evaluations)
    hv_enabled = hv_stop_config is not None and algorithm_name == "nsgaii"
    if hv_enabled:
        hv_termination = dict(hv_stop_config)
        hv_termination["max_evaluations"] = config.max_evaluations
        termination = ("hv", hv_termination)

    validate_problem(problem)

    ensure_dir(output_dir)
    exec_result = execute_algorithm(
        algorithm,
        problem,
        termination=termination,
        seed=config.seed,
        eval_backend=eval_backend,
        live_viz=live_viz,
    )
    payload = exec_result.payload
    total_time_ms = exec_result.elapsed_ms
    F = payload["F"]
    archive = payload.get("archive")
    actual_evaluations = int(payload.get("evaluations", config.max_evaluations))
    termination_reason = "max_evaluations"
    if hv_enabled and payload.get("hv_reached"):
        termination_reason = "hv_threshold"

    metrics = build_metrics(
        algorithm_name, engine_name, total_time_ms, actual_evaluations, F
    )
    metrics["termination"] = termination_reason
    metrics["eval_backend"] = getattr(config, "eval_backend", "serial")
    metrics["n_workers"] = getattr(config, "n_workers", None)
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
    print_run_results(metrics)
    metrics["output_dir"] = output_dir
    persist_run_outputs(
        output_dir=output_dir,
        selection=selection,
        algorithm_name=algorithm_name,
        engine_name=engine_name,
        cfg_data=cfg_data,
        metrics=metrics,
        payload=payload,
        total_time_ms=total_time_ms,
        hv_stop_config=hv_stop_config,
        config_source=config_source,
        selection_pressure=selection_pressure,
        external_archive_size=external_archive_size,
        encoding=getattr(problem, "encoding", getattr(selection.spec, "encoding", "continuous")),
        problem_override=problem_override,
        autodiff_info=autodiff_info,
        config=config,
        kernel_backend=kernel_backend,
        project_root=PROJECT_ROOT,
        nsgaii_variation=nsgaii_variation,
        moead_variation=moead_variation,
        smsemoa_variation=smsemoa_variation,
        nsga3_variation=nsga3_variation,
    )

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
    spea2_variation: dict | None = None,
    ibea_variation: dict | None = None,
    smpso_variation: dict | None = None,
    include_external: bool = False,
    config_source: str | None = None,
    problem_override: dict | None = None,
    track_genealogy: bool = False,
    autodiff_constraints: bool = False,
):
    from vamos.foundation.core import external  # local import to keep runner decoupled
    from vamos.ux.visualization import plotting

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
                spea2_variation=getattr(args, "spea2_variation", None),
                ibea_variation=getattr(args, "ibea_variation", None),
                smpso_variation=getattr(args, "smpso_variation", None),
                hv_stop_config=hv_stop_config if algorithm_name == "nsgaii" else None,
                config_source=config_source,
                problem_override=problem_override,
                track_genealogy=getattr(args, "track_genealogy", False) and algorithm_name == "nsgaii",
                autodiff_constraints=getattr(args, "autodiff_constraints", False),
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
                spea2_variation=getattr(args, "spea2_variation", None),
                ibea_variation=getattr(args, "ibea_variation", None),
                smpso_variation=getattr(args, "smpso_variation", None),
                hv_stop_config=hv_stop_config if algorithm_name == "nsgaii" else None,
                config_source=config_source,
                problem_override=problem_override,
                track_genealogy=getattr(args, "track_genealogy", False) and algorithm_name == "nsgaii",
                autodiff_constraints=getattr(args, "autodiff_constraints", False),
            )
            results.append(metrics)

    for algorithm_name in external_algorithms:
        metrics = external.run_external(
            algorithm_name,
            problem_selection,
            use_native_problem=use_native_external_problem,
            config=config,
            make_metrics=_make_metrics,
            print_banner=lambda problem, selection, label, backend: print_run_banner(
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
            eval_backend=override.get("eval_backend", getattr(config, "eval_backend", "serial")),
            n_workers=override.get("n_workers", getattr(config, "n_workers", None)),
            live_viz=override.get("live_viz", getattr(config, "live_viz", False)),
            live_viz_interval=override.get("live_viz_interval", getattr(config, "live_viz_interval", 5)),
            live_viz_max_points=override.get("live_viz_max_points", getattr(config, "live_viz_max_points", 1000)),
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
        effective_args.eval_backend = override.get("eval_backend", args.eval_backend)
        effective_args.n_workers = override.get("n_workers", args.n_workers)
        effective_args.live_viz = override.get("live_viz", args.live_viz)
        effective_args.live_viz_interval = override.get("live_viz_interval", args.live_viz_interval)
        effective_args.live_viz_max_points = override.get("live_viz_max_points", args.live_viz_max_points)
        effective_args.track_genealogy = override.get("track_genealogy", getattr(args, "track_genealogy", False))
        effective_args.autodiff_constraints = override.get("autodiff_constraints", getattr(args, "autodiff_constraints", False))
        effective_args.nsgaii_variation = merge_variation_overrides(base_variation, override.get("nsgaii"))
        effective_args.moead_variation = merge_variation_overrides(getattr(args, "moead_variation", None), override.get("moead"))
        effective_args.smsemoa_variation = merge_variation_overrides(getattr(args, "smsemoa_variation", None), override.get("smsemoa"))
        effective_args.nsga3_variation = merge_variation_overrides(getattr(args, "nsga3_variation", None), override.get("nsga3"))
        effective_args.spea2_variation = merge_variation_overrides(getattr(args, "spea2_variation", None), override.get("spea2"))
        effective_args.ibea_variation = merge_variation_overrides(getattr(args, "ibea_variation", None), override.get("ibea"))
        effective_args.smpso_variation = merge_variation_overrides(getattr(args, "smpso_variation", None), override.get("smpso"))
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
            spea2_variation=effective_args.spea2_variation,
            ibea_variation=effective_args.ibea_variation,
            smpso_variation=effective_args.smpso_variation,
            include_external=effective_args.include_external,
            config_source=config_source,
            problem_override=override,
            track_genealogy=effective_args.track_genealogy,
            autodiff_constraints=effective_args.autodiff_constraints,
        )
