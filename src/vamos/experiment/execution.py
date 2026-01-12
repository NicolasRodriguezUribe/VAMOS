from __future__ import annotations

import logging
from argparse import Namespace
from importlib.resources import as_file
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from vamos.foundation.encoding import normalize_encoding
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
from vamos.experiment.runner_abstractions import resolve_evaluator, resolve_termination
from vamos.foundation.metrics.hypervolume import hypervolume
from vamos.foundation.problem.registry import ProblemSelection
from vamos.foundation.observer import Observer, RunContext
from vamos.experiment.runner_utils import run_output_dir, validate_problem
from vamos.experiment.observers.console import ConsoleObserver
from vamos.experiment.observers.storage import StorageObserver
from vamos.hooks import (
    HookManager,
    HookManagerConfig,
    LiveVisualization,
)
from vamos.hooks.config_parse import parse_stopping_archive


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


Metrics = dict[str, Any]
VariationConfig = dict[str, Any]


class CompositeObserver(Observer):
    """Fans out events to multiple observers."""

    def __init__(self, observers: list[Observer]):
        self.observers = [o for o in observers if o is not None]

    def on_start(self, ctx: RunContext) -> None:
        for obs in self.observers:
            obs.on_start(ctx)

    def on_generation(
        self,
        generation: int,
        F: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        stats: Optional[dict[str, Any]] = None,
    ) -> None:
        for obs in self.observers:
            obs.on_generation(generation, F, X, stats)

    def on_end(
        self,
        final_F: Optional[np.ndarray] = None,
        final_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        for obs in self.observers:
            obs.on_end(final_F, final_stats)

    def should_stop(self) -> bool:
        # Check if any observer requests stopping (e.g. Hooks)
        # Note: Observer protocol doesn't strictly have should_stop, but HookManager does.
        # We handle this specifically for observers that support it.
        for obs in self.observers:
            if hasattr(obs, "should_stop") and obs.should_stop():
                return True
        return False


class _LiveVizAdapter:
    """Bridge algorithm live-viz callbacks to observer on_generation events."""

    def __init__(self, observer: CompositeObserver):
        self._observer = observer

    def on_start(self, ctx: RunContext) -> None:
        return None

    def on_generation(
        self,
        generation: int,
        F: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        stats: Optional[dict[str, Any]] = None,
    ) -> None:
        self._observer.on_generation(generation, F, X, stats)

    def on_end(
        self,
        final_F: Optional[np.ndarray] = None,
        final_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        return None

    def should_stop(self) -> bool:
        return self._observer.should_stop()


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
    evaluator: Any | None = None,
    termination: tuple[str, Any] | None = None,
    config_source: str | None = None,
    config_spec: dict[str, Any] | None = None,
    problem_override: dict[str, Any] | None = None,
    track_genealogy: bool = False,
    autodiff_constraints: bool = False,
    live_viz: LiveVisualization | None = None,
) -> Metrics:
    problem = problem or selection.instantiate()

    # 1. Build Context
    ctx = RunContext(
        problem=problem,
        algorithm=algorithm,
        config=config,
        selection=selection,
        algorithm_name=algorithm_name,
        engine_name=engine_name,
    )

    output_dir = run_output_dir(selection, algorithm_name, engine_name, config.seed, config)
    ensure_dir(output_dir)

    termination_spec = resolve_termination(
        termination,
        config,
        hv_stop_config=hv_stop_config,
        algorithm_name=algorithm_name,
    )
    termination_kind, termination_payload = termination_spec
    hv_enabled = termination_kind == "hv"
    hv_termination = termination_payload if hv_enabled else None
    if hv_enabled and isinstance(hv_termination, dict):
        hv_stop_config = hv_termination

    # 2. Build Observers
    observers: list[Observer] = []

    # Console
    observers.append(ConsoleObserver())

    # Storage
    # Collect variations for metadata
    variations = {
        "nsgaii_variation": nsgaii_variation,
        "moead_variation": moead_variation,
        "smsemoa_variation": smsemoa_variation,
        "nsgaiii_variation": nsgaiii_variation,
        "spea2_variation": spea2_variation,
        "ibea_variation": ibea_variation,
        "smpso_variation": smpso_variation,
    }

    storage = StorageObserver(
        output_dir=output_dir,
        project_root=_project_root(),
        config_source=config_source,
        problem_override=problem_override,
        hv_stop_config=hv_stop_config,
        selection_pressure=selection_pressure,
        external_archive_size=external_archive_size,
        variations=variations,
    )
    observers.append(storage)

    # Hooks
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
                observers.append(hook_mgr)
        except Exception:
            hook_mgr = None

    # User Viz (LiveVisualization uses RunContext)
    if live_viz:
        observers.append(live_viz)

    main_observer = CompositeObserver(observers)

    # 3. Notify Start
    main_observer.on_start(ctx)

    autodiff_info: dict[str, Any] | None = None
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

    eval_strategy = resolve_evaluator(evaluator, config)

    kernel_backend = getattr(algorithm, "kernel", None)

    if hv_enabled and not isinstance(hv_termination, dict):
        hv_enabled = False
        hv_termination = None

    validate_problem(problem)
    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    if encoding == "permutation" and algorithm_name != "nsgaii":
        raise ValueError("Permutation problems are only supported by NSGA-II.")

    # 4. Execute
    exec_result = execute_algorithm(
        algorithm,
        problem,
        termination=termination_spec,
        seed=config.seed,
        eval_strategy=eval_strategy,
        live_viz=_LiveVizAdapter(main_observer),
    )

    payload = exec_result.payload
    total_time_ms = exec_result.elapsed_ms
    F = payload["F"]
    actual_evaluations = int(payload.get("evaluations", config.max_evaluations))

    # Termination reason determination
    termination_reason = "max_evaluations"
    if main_observer.should_stop():  # Check hooks via composite
        termination_reason = "hv_convergence"
    if hv_enabled and payload.get("hv_reached"):
        termination_reason = "hv_threshold"

    # 5. Build Final Stats / Metrics
    spread = None
    if F.size and F.shape[1] >= 1:
        spread = np.ptp(F[:, 0])
    evals_per_sec = actual_evaluations / max(1e-9, total_time_ms / 1000.0)

    metrics = {
        "algorithm": algorithm_name,
        "engine": engine_name,
        "time_ms": total_time_ms,
        "evaluations": actual_evaluations,
        "evals_per_sec": evals_per_sec,
        "spread": spread,
        "F": F,
        "X": payload.get("X"),
        "termination": termination_reason,
        "eval_strategy": getattr(config, "eval_strategy", "serial"),
        "n_workers": getattr(config, "n_workers", None),
        "config": cfg_data,
        "_kernel_backend": kernel_backend,
        "backend_device": kernel_backend.device() if kernel_backend else "external",
        "backend_capabilities": sorted(set(kernel_backend.capabilities())) if kernel_backend else [],
        "output_dir": output_dir,  # Needed?
        "payload": payload,  # Pass payload for StorageObserver to extract archives etc.
        "autodiff_info": autodiff_info,
    }

    if hook_mgr is not None:
        metrics["hook_metadata"] = hook_mgr.metadata_payload()

    if hv_enabled and hv_termination:
        metrics["hv_threshold_fraction"] = hv_termination.get("threshold_fraction")
        metrics["hv_reference_point"] = hv_termination.get("reference_point")
        metrics["hv_reference_front"] = hv_termination.get("reference_front_path")

    # 6. Notify End
    main_observer.on_end(F, metrics)

    return metrics  # Return metrics for benchmark aggregation


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
        from vamos.experiment.external.reporting import build_metrics, print_run_banner, print_run_results

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
