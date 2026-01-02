from __future__ import annotations

import logging
from argparse import Namespace
from importlib.resources import as_file
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from vamos.foundation.core.execution import execute_algorithm
from vamos.foundation.core.experiment_config import (
    ENABLED_ALGORITHMS,
    EXPERIMENT_BACKENDS,
    EXTERNAL_ALGORITHM_NAMES,
    OPTIONAL_ALGORITHMS,
    ExperimentConfig,
)
from vamos.foundation.core.hv_stop import compute_hv_reference
from vamos.foundation.core.io_utils import ensure_dir
from vamos.foundation.data import weight_path
from vamos.foundation.eval.backends import resolve_eval_backend
from vamos.foundation.metrics.hypervolume import hypervolume
from vamos.foundation.problem.resolver import ProblemSelection
from vamos.experiment.runner_output import (
    build_metrics,
    persist_run_outputs,
    print_run_banner,
    print_run_results,
)
from vamos.experiment.runner_utils import run_output_dir, validate_problem
from vamos.hooks import (
    CompositeLiveVisualization,
    HookManager,
    HookManagerConfig,
    LiveVisualization,
    NoOpLiveVisualization,
)
from vamos.hooks.config_parse import parse_stopping_archive


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


Metrics = dict[str, Any]
VariationConfig = dict[str, Any]


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
    algorithm: Any,
    cfg_data: dict[str, Any],
    problem: Any | None = None,
    external_archive_size: int | None = None,
    archive_type: str = "hypervolume",
    selection_pressure: int = 2,
    nsgaii_variation: VariationConfig | None = None,
    moead_variation: VariationConfig | None = None,
    smsemoa_variation: VariationConfig | None = None,
    nsgaiii_variation: VariationConfig | None = None,
    spea2_variation: VariationConfig | None = None,
    ibea_variation: VariationConfig | None = None,
    smpso_variation: VariationConfig | None = None,
    hv_stop_config: dict[str, Any] | None = None,
    config_source: str | None = None,
    config_spec: dict[str, Any] | None = None,
    problem_override: dict[str, Any] | None = None,
    track_genealogy: bool = False,
    autodiff_constraints: bool = False,
    live_viz: LiveVisualization | None = None,
) -> Metrics:
    problem = problem or selection.instantiate()
    display_algo = algorithm_name.upper()
    print_run_banner(problem, selection, display_algo, engine_name, config)

    autodiff_info = None
    if autodiff_constraints:
        autodiff_info = {"status": "unavailable"}
        try:
            from vamos.foundation.constraints.autodiff import build_jax_constraint_functions

            cm = getattr(problem, "constraint_model", None)
            if callable(cm):
                cm = cm()
            if cm is not None:
                build_jax_constraint_functions(cm)
                autodiff_info = {
                    "status": "ok",
                    "n_constraints": len(getattr(cm, "constraints", []) or []),
                }
            else:
                autodiff_info = {"status": "no_constraint_model"}
        except Exception as exc:
            autodiff_info = {"status": "error", "message": str(exc)}

    eval_backend = resolve_eval_backend(
        getattr(config, "eval_backend", "serial"),
        n_workers=getattr(config, "n_workers", None),
    )
    output_dir = run_output_dir(selection, algorithm_name, engine_name, config.seed, config)
    visualizer = live_viz or NoOpLiveVisualization()
    hook_mgr = None
    if isinstance(config_spec, dict):
        try:
            hook_cfg = parse_stopping_archive(config_spec, problem_key=selection.spec.key)
            if hook_cfg.get("stopping_enabled") or hook_cfg.get("archive_enabled"):
                hook_mgr = HookManager(
                    out_dir=Path(output_dir),
                    cfg=HookManagerConfig(
                        stopping_enabled=hook_cfg["stopping_enabled"],
                        stop_cfg=hook_cfg["stop_cfg"],
                        archive_enabled=hook_cfg["archive_enabled"],
                        archive_cfg=hook_cfg["archive_cfg"],
                        hv_ref_point=hook_cfg.get("hv_ref_point"),
                    ),
                )
                if live_viz is None:
                    visualizer = hook_mgr
                else:
                    visualizer = CompositeLiveVisualization([hook_mgr, visualizer])
        except Exception:
            hook_mgr = None

    kernel_backend = getattr(algorithm, "kernel", None)

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
        live_viz=visualizer,
    )
    payload = exec_result.payload
    total_time_ms = exec_result.elapsed_ms
    F = payload["F"]
    actual_evaluations = int(payload.get("evaluations", config.max_evaluations))
    termination_reason = "max_evaluations"
    if hook_mgr is not None and hook_mgr.should_stop():
        termination_reason = "hv_convergence"
    if hv_enabled and payload.get("hv_reached"):
        termination_reason = "hv_threshold"

    metrics: Metrics = build_metrics(algorithm_name, engine_name, total_time_ms, actual_evaluations, F)
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
        project_root=_project_root(),
        nsgaii_variation=nsgaii_variation,
        moead_variation=moead_variation,
        smsemoa_variation=smsemoa_variation,
        nsgaiii_variation=nsgaiii_variation,
        hook_mgr=hook_mgr,
    )

    _logger().info("Results stored in: %s", output_dir)
    _logger().info("%s", "=" * 80)

    return metrics


def _print_summary(results: Iterable[Metrics], hv_ref_point: np.ndarray) -> None:
    _logger().info("Experiment summary")
    _logger().info("%s", "-" * 80)
    header = f"{'Algo':<12} {'Backend':<10} {'Time (ms)':>12} {'Eval/s':>12} {'HV':>12} {'Spread f1':>12}"
    _logger().info("%s", header)
    _logger().info("%s", "-" * len(header))
    for res in results:
        spread = res["spread"]
        spread_txt = f"{spread:.6f}" if spread is not None else "-"
        hv_txt = f"{res['hv']:.6f}" if res.get("hv") is not None else "-"
        _logger().info(
            "%s",
            f"{res['algorithm']:<12} {res['engine']:<10} {res['time_ms']:>12.2f} "
            f"{res['evals_per_sec']:>12.1f} {hv_txt:>12} {spread_txt:>12}",
        )
    ref_txt = np.array2string(hv_ref_point, precision=3, suppress_small=True)
    _logger().info("Hypervolume reference point: %s", ref_txt)


def execute_problem_suite(
    args: Namespace,
    problem_selection: ProblemSelection,
    config: ExperimentConfig,
    *,
    run_single_fn: Callable[..., Metrics],
    hv_stop_config: dict[str, Any] | None = None,
    nsgaii_variation: VariationConfig | None = None,
    spea2_variation: VariationConfig | None = None,
    ibea_variation: VariationConfig | None = None,
    smpso_variation: VariationConfig | None = None,
    include_external: bool = False,
    config_source: str | None = None,
    config_spec: dict[str, Any] | None = None,
    problem_override: dict[str, Any] | None = None,
    track_genealogy: bool = False,
    autodiff_constraints: bool = False,
    live_viz_factory: Callable[..., LiveVisualization | None] | None = None,
    plotter: Callable[..., Any] | None = None,
) -> None:
    from vamos.experiment import external  # local import to keep runner decoupled

    engines: Iterable[str] = EXPERIMENT_BACKENDS if args.experiment == "backends" else (args.engine,)
    algorithms = list(ENABLED_ALGORITHMS) if args.algorithm == "both" else [args.algorithm]
    use_native_external_problem = args.external_problem_source == "native"

    if include_external and problem_selection.spec.key != "zdt1":
        _logger().info("External baselines are currently available only for ZDT1; skipping external runs.")
        include_external = False

    if include_external:
        for ext in EXTERNAL_ALGORITHM_NAMES:
            if ext not in algorithms:
                algorithms.append(ext)

    internal_algorithms = [a for a in algorithms if a in ENABLED_ALGORITHMS]
    optional_algorithms = [a for a in algorithms if a in OPTIONAL_ALGORITHMS]
    external_algorithms = [a for a in algorithms if a in EXTERNAL_ALGORITHM_NAMES]

    results: list[Metrics] = []
    for engine in engines:
        for algorithm_name in internal_algorithms:
            live_viz = None
            if live_viz_factory is not None:
                live_viz = live_viz_factory(
                    problem_selection,
                    algorithm_name,
                    engine,
                    config,
                )
            metrics = run_single_fn(
                engine,
                algorithm_name,
                problem_selection,
                config,
                external_archive_size=args.external_archive_size,
                selection_pressure=args.selection_pressure,
                nsgaii_variation=nsgaii_variation,
                moead_variation=getattr(args, "moead_variation", None),
                smsemoa_variation=getattr(args, "smsemoa_variation", None),
                nsgaiii_variation=getattr(args, "nsgaiii_variation", None),
                spea2_variation=getattr(args, "spea2_variation", None),
                ibea_variation=getattr(args, "ibea_variation", None),
                smpso_variation=getattr(args, "smpso_variation", None),
                hv_stop_config=hv_stop_config if algorithm_name == "nsgaii" else None,
                config_source=config_source,
                config_spec=config_spec,
                problem_override=problem_override,
                track_genealogy=(getattr(args, "track_genealogy", False) and algorithm_name == "nsgaii"),
                autodiff_constraints=getattr(args, "autodiff_constraints", False),
                live_viz=live_viz,
            )
            results.append(metrics)
        for algorithm_name in optional_algorithms:
            live_viz = None
            if live_viz_factory is not None:
                live_viz = live_viz_factory(
                    problem_selection,
                    algorithm_name,
                    engine,
                    config,
                )
            metrics = run_single_fn(
                engine,
                algorithm_name,
                problem_selection,
                config,
                external_archive_size=args.external_archive_size,
                selection_pressure=args.selection_pressure,
                nsgaii_variation=nsgaii_variation,
                moead_variation=getattr(args, "moead_variation", None),
                smsemoa_variation=getattr(args, "smsemoa_variation", None),
                nsgaiii_variation=getattr(args, "nsgaiii_variation", None),
                spea2_variation=getattr(args, "spea2_variation", None),
                ibea_variation=getattr(args, "ibea_variation", None),
                smpso_variation=getattr(args, "smpso_variation", None),
                hv_stop_config=hv_stop_config if algorithm_name == "nsgaii" else None,
                config_source=config_source,
                config_spec=config_spec,
                problem_override=problem_override,
                track_genealogy=(getattr(args, "track_genealogy", False) and algorithm_name == "nsgaii"),
                autodiff_constraints=getattr(args, "autodiff_constraints", False),
                live_viz=live_viz,
            )
            results.append(metrics)

    for algorithm_name in external_algorithms:
        metrics = external.run_external(
            algorithm_name,
            problem_selection,
            use_native_problem=use_native_external_problem,
            config=config,
            make_metrics=build_metrics,
            print_banner=lambda problem, selection, label, backend: print_run_banner(problem, selection, label, backend, config),
            print_results=print_run_results,
        )
        if metrics is not None:
            results.append(metrics)

    if not results:
        _logger().info("No runs were executed. Check algorithm selection or install missing dependencies.")
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
        _logger().info("Hypervolume (reference %s): %.6f", ref_txt, hv_val)
    else:
        _print_summary(results, hv_ref_point)

    if plotter is not None:
        plotter(results, problem_selection, output_root=config.output_root, title=config.title)
