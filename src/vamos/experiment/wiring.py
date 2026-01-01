from __future__ import annotations

import logging
import os
from argparse import Namespace
from copy import deepcopy
from typing import Any, Callable, Iterable, Sequence

from vamos.engine.algorithm.factory import build_algorithm
from vamos.engine.config.loader import load_experiment_spec
from vamos.engine.config.variation import merge_variation_overrides, normalize_variation_config
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.core.hv_stop import build_hv_stop_config
from vamos.foundation.problem.registry import ProblemSelection, make_problem_selection
from vamos.foundation.problem.resolver import resolve_problem_selections
from vamos.hooks import LiveVisualization
from vamos.ux.visualization import plotting
from vamos.ux.visualization.live_viz import LiveParetoPlot

from .execution import execute_problem_suite, run_single as execute_run_single
from .runner_utils import run_output_dir
from .study.runner import StudyRunner, StudyResult, StudyTask

logger = logging.getLogger(__name__)
VariationConfig = dict[str, Any]


def normalize_variations(
    *,
    nsgaii_variation: VariationConfig | None = None,
    moead_variation: VariationConfig | None = None,
    smsemoa_variation: VariationConfig | None = None,
    nsgaiii_variation: VariationConfig | None = None,
    spea2_variation: VariationConfig | None = None,
    ibea_variation: VariationConfig | None = None,
    smpso_variation: VariationConfig | None = None,
) -> tuple[
    VariationConfig | None,
    VariationConfig | None,
    VariationConfig | None,
    VariationConfig | None,
    VariationConfig | None,
    VariationConfig | None,
    VariationConfig | None,
]:
    return (
        normalize_variation_config(nsgaii_variation),
        normalize_variation_config(moead_variation),
        normalize_variation_config(smsemoa_variation),
        normalize_variation_config(nsgaiii_variation),
        normalize_variation_config(spea2_variation),
        normalize_variation_config(ibea_variation),
        normalize_variation_config(smpso_variation),
    )


def build_algorithm_from_spec(
    algorithm_name: str,
    engine_name: str,
    problem: Any,
    config: ExperimentConfig,
    *,
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
    track_genealogy: bool = False,
) -> tuple[Any, dict[str, Any]]:
    return build_algorithm(
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
        nsgaiii_variation=nsgaiii_variation,
        spea2_variation=spea2_variation,
        ibea_variation=ibea_variation,
        smpso_variation=smpso_variation,
        track_genealogy=track_genealogy,
    )


def run_single(
    engine_name: str,
    algorithm_name: str,
    selection: ProblemSelection,
    config: ExperimentConfig,
    *,
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
) -> dict[str, Any]:
    problem = selection.instantiate()
    (
        nsgaii_variation,
        moead_variation,
        smsemoa_variation,
        nsgaiii_variation,
        spea2_variation,
        ibea_variation,
        smpso_variation,
    ) = normalize_variations(
        nsgaii_variation=nsgaii_variation,
        moead_variation=moead_variation,
        smsemoa_variation=smsemoa_variation,
        nsgaiii_variation=nsgaiii_variation,
        spea2_variation=spea2_variation,
        ibea_variation=ibea_variation,
        smpso_variation=smpso_variation,
    )
    algorithm, cfg_data = build_algorithm_from_spec(
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
        nsgaiii_variation=nsgaiii_variation,
        spea2_variation=spea2_variation,
        ibea_variation=ibea_variation,
        smpso_variation=smpso_variation,
        track_genealogy=track_genealogy,
    )
    return execute_run_single(
        engine_name,
        algorithm_name,
        selection,
        config,
        algorithm=algorithm,
        cfg_data=cfg_data,
        problem=problem,
        external_archive_size=external_archive_size,
        archive_type=archive_type,
        selection_pressure=selection_pressure,
        nsgaii_variation=nsgaii_variation,
        moead_variation=moead_variation,
        smsemoa_variation=smsemoa_variation,
        nsgaiii_variation=nsgaiii_variation,
        spea2_variation=spea2_variation,
        ibea_variation=ibea_variation,
        smpso_variation=smpso_variation,
        hv_stop_config=hv_stop_config,
        config_source=config_source,
        config_spec=config_spec,
        problem_override=problem_override,
        track_genealogy=track_genealogy,
        autodiff_constraints=autodiff_constraints,
        live_viz=live_viz,
    )


def run_from_args(
    args: Namespace,
    config: ExperimentConfig,
    *,
    live_viz_factory: Callable[..., LiveVisualization | None] | None = None,
    plotter: Callable[..., Any] | None = None,
) -> None:
    selections = list(resolve_problem_selections(args))
    multiple = len(selections) > 1
    base_variation = getattr(args, "nsgaii_variation", None)
    overrides: dict[str, Any] = getattr(args, "problem_overrides", {}) or {}
    config_source = getattr(args, "config_path", None)
    config_spec: dict[str, Any] | None = None
    if config_source:
        try:
            config_spec = load_experiment_spec(config_source)
        except Exception:
            config_spec = None

    for idx, selection in enumerate(selections, start=1):
        override: dict[str, Any] = overrides.get(selection.spec.key, {}) or {}
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
            offspring_population_size=override.get("offspring_population_size", config.offspring_population_size),
            max_evaluations=override.get("max_evaluations", config.max_evaluations),
            seed=override.get("seed", config.seed),
            eval_backend=override.get("eval_backend", getattr(config, "eval_backend", "serial")),
            n_workers=override.get("n_workers", getattr(config, "n_workers", None)),
            live_viz=override.get("live_viz", getattr(config, "live_viz", False)),
            live_viz_interval=override.get("live_viz_interval", getattr(config, "live_viz_interval", 5)),
            live_viz_max_points=override.get("live_viz_max_points", getattr(config, "live_viz_max_points", 1000)),
        )
        effective_args = deepcopy(args)
        for key in (
            "algorithm",
            "engine",
            "experiment",
            "include_external",
            "external_problem_source",
        ):
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
        effective_args.nsgaiii_variation = merge_variation_overrides(getattr(args, "nsgaiii_variation", None), override.get("nsgaiii"))
        effective_args.spea2_variation = merge_variation_overrides(getattr(args, "spea2_variation", None), override.get("spea2"))
        effective_args.ibea_variation = merge_variation_overrides(getattr(args, "ibea_variation", None), override.get("ibea"))
        effective_args.smpso_variation = merge_variation_overrides(getattr(args, "smpso_variation", None), override.get("smpso"))
        effective_args.effective_problem_override = override

        (
            effective_args.nsgaii_variation,
            effective_args.moead_variation,
            effective_args.smsemoa_variation,
            effective_args.nsgaiii_variation,
            effective_args.spea2_variation,
            effective_args.ibea_variation,
            effective_args.smpso_variation,
        ) = normalize_variations(
            nsgaii_variation=effective_args.nsgaii_variation,
            moead_variation=effective_args.moead_variation,
            smsemoa_variation=effective_args.smsemoa_variation,
            nsgaiii_variation=effective_args.nsgaiii_variation,
            spea2_variation=effective_args.spea2_variation,
            ibea_variation=effective_args.ibea_variation,
            smpso_variation=effective_args.smpso_variation,
        )

        if multiple:
            logger.info("%s", "\n" + "#" * 80)
            logger.info(
                "Problem %s/%s: %s (%s)",
                idx,
                len(selections),
                effective_selection.spec.label,
                effective_selection.spec.key,
            )
            logger.info("%s", "#" * 80 + "\n")

        hv_stop_config = None
        if effective_args.hv_threshold is not None:
            hv_stop_config = build_hv_stop_config(
                effective_args.hv_threshold,
                effective_args.hv_reference_front,
                effective_selection.spec.key,
            )
        nsgaii_variation = getattr(effective_args, "nsgaii_variation", None)
        execute_problem_suite(
            effective_args,
            effective_selection,
            effective_config,
            run_single_fn=run_single,
            hv_stop_config=hv_stop_config,
            nsgaii_variation=nsgaii_variation,
            spea2_variation=effective_args.spea2_variation,
            ibea_variation=effective_args.ibea_variation,
            smpso_variation=effective_args.smpso_variation,
            include_external=effective_args.include_external,
            config_source=config_source,
            config_spec=config_spec,
            problem_override=override,
            track_genealogy=effective_args.track_genealogy,
            autodiff_constraints=effective_args.autodiff_constraints,
            live_viz_factory=live_viz_factory,
            plotter=plotter,
        )


# --- Experiment-layer convenience wrappers ---


def _build_live_viz(
    selection: ProblemSelection,
    algorithm: str,
    engine: str,
    config: ExperimentConfig,
) -> LiveVisualization | None:
    if not getattr(config, "live_viz", False):
        return None
    output_dir = run_output_dir(selection, algorithm, engine, config.seed, config)
    return LiveParetoPlot(
        update_interval=getattr(config, "live_viz_interval", 5),
        max_points=getattr(config, "live_viz_max_points", 1000),
        save_final_path=os.path.join(output_dir, "live_pareto.png"),
        title=f"{selection.spec.label} (live)",
    )


def run_experiment(
    *,
    algorithm: str,
    problem: str,
    engine: str = "numpy",
    config: ExperimentConfig | None = None,
    n_var: int | None = None,
    n_obj: int | None = None,
    selection_pressure: int = 2,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Execute a single problem/algorithm/engine combination.

    Parameters mirror the CLI; additional keyword args are passed to `run_single`
    (e.g., `external_archive_size`, `hv_stop_config`, `track_genealogy`).
    """
    cfg = config or ExperimentConfig()
    selection = make_problem_selection(problem, n_var=n_var, n_obj=n_obj)
    live_viz = kwargs.pop("live_viz", None)
    if live_viz is None:
        live_viz = _build_live_viz(selection, algorithm, engine, cfg)
    return run_single(
        engine,
        algorithm,
        selection,
        cfg,
        selection_pressure=selection_pressure,
        live_viz=live_viz,
        **kwargs,
    )


def run_experiments_from_args(args: Namespace, config: ExperimentConfig) -> None:
    """
    Entry point used by the CLI to execute one or more runs defined by parsed args.
    """
    run_from_args(
        args,
        config,
        live_viz_factory=_build_live_viz,
        plotter=plotting.plot_pareto_front,
    )


def run_study(
    tasks: Iterable[StudyTask],
    *,
    config_overrides: dict[str, Any] | None = None,
    mirror_output_roots: Sequence[str] | None = ("results",),
) -> list[StudyResult]:
    runner = StudyRunner(mirror_output_roots=mirror_output_roots)
    overrides: dict[str, Any] = config_overrides or {}
    if overrides:
        adjusted: list[StudyTask] = []
        for task in tasks:
            merged: dict[str, Any] = dict(task.config_overrides or {})
            merged.update({k: v for k, v in overrides.items() if v is not None})
            adjusted.append(
                StudyTask(
                    algorithm=task.algorithm,
                    engine=task.engine,
                    problem=task.problem,
                    n_var=task.n_var,
                    n_obj=task.n_obj,
                    seed=task.seed,
                    selection_pressure=task.selection_pressure,
                    external_archive_size=task.external_archive_size,
                    archive_type=task.archive_type,
                    nsgaii_variation=task.nsgaii_variation,
                    config_overrides=merged,
                )
            )
        tasks = adjusted
    return runner.run(list(tasks), run_single_fn=run_single)
