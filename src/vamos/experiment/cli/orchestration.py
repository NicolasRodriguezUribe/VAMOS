from __future__ import annotations

import logging
from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy
from typing import TypeVar, cast

from vamos.archive import ExternalArchiveConfig
from vamos.engine.config.variation import VariationOverrides, merge_variation_overrides
from vamos.engine.config.spec import ExperimentSpec, ProblemOverrides, SpecBlock
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.core.hv_stop import build_hv_stop_config
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.problem.resolver import resolve_problem_selections
from vamos.hooks import LiveVisualization

from vamos.experiment.execution import execute_problem_suite
from vamos.experiment.services.config import normalize_variations
from vamos.experiment.services.orchestrator import run_single


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


T = TypeVar("T")


def _override_value(override: SpecBlock, key: str, default: T) -> T:
    return cast(T, override.get(key, default))


def _override_mapping(override: SpecBlock, key: str) -> VariationOverrides | None:
    value = override.get(key)
    return cast(VariationOverrides, value) if isinstance(value, dict) else None


def run_from_args(
    args: Namespace,
    config: ExperimentConfig,
    *,
    live_viz_factory: Callable[..., LiveVisualization | None] | None = None,
    plotter: Callable[..., object] | None = None,
) -> None:
    selections = list(resolve_problem_selections(args))
    multiple = len(selections) > 1
    base_variation = getattr(args, "nsgaii_variation", None)
    overrides: ProblemOverrides = getattr(args, "problem_overrides", {}) or {}
    config_source = getattr(args, "config_path", None)
    config_spec: ExperimentSpec | None = getattr(args, "config_spec", None)

    for idx, selection in enumerate(selections, start=1):
        override_raw = overrides.get(selection.spec.key)
        override: SpecBlock = override_raw if isinstance(override_raw, dict) else {}
        effective_selection = selection
        if override.get("n_var") is not None or override.get("n_obj") is not None:
            effective_selection = make_problem_selection(
                selection.spec.key,
                n_var=_override_value(override, "n_var", cast(int | None, selection.n_var)),
                n_obj=_override_value(override, "n_obj", cast(int | None, selection.n_obj)),
            )
        effective_config = ExperimentConfig(
            title=_override_value(override, "title", config.title),
            output_root=_override_value(override, "output_root", config.output_root),
            population_size=_override_value(override, "population_size", config.population_size),
            offspring_population_size=_override_value(override, "offspring_population_size", config.offspring_population_size),
            max_evaluations=_override_value(override, "max_evaluations", config.max_evaluations),
            seed=_override_value(override, "seed", config.seed),
            eval_strategy=_override_value(override, "eval_strategy", getattr(config, "eval_strategy", "serial")),
            n_workers=_override_value(override, "n_workers", getattr(config, "n_workers", None)),
            live_viz=_override_value(override, "live_viz", getattr(config, "live_viz", False)),
            live_viz_interval=_override_value(override, "live_viz_interval", getattr(config, "live_viz_interval", 5)),
            live_viz_max_points=_override_value(override, "live_viz_max_points", getattr(config, "live_viz_max_points", 1000)),
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
        _ea_override = override.get("external_archive_size")
        if _ea_override is not None:
            effective_args.external_archive = ExternalArchiveConfig(capacity=int(str(_ea_override)))
        else:
            effective_args.external_archive = getattr(args, "external_archive", None)
        effective_args.hv_threshold = override.get("hv_threshold", args.hv_threshold)
        effective_args.hv_reference_front = override.get("hv_reference_front", args.hv_reference_front)
        effective_args.n_var = override.get("n_var", args.n_var)
        effective_args.n_obj = override.get("n_obj", args.n_obj)
        effective_args.eval_strategy = override.get("eval_strategy", args.eval_strategy)
        effective_args.n_workers = override.get("n_workers", args.n_workers)
        effective_args.live_viz = override.get("live_viz", args.live_viz)
        effective_args.live_viz_interval = override.get("live_viz_interval", args.live_viz_interval)
        effective_args.live_viz_max_points = override.get("live_viz_max_points", args.live_viz_max_points)
        effective_args.track_genealogy = override.get("track_genealogy", getattr(args, "track_genealogy", False))
        effective_args.autodiff_constraints = override.get("autodiff_constraints", getattr(args, "autodiff_constraints", False))
        effective_args.nsgaii_variation = merge_variation_overrides(base_variation, _override_mapping(override, "nsgaii"))
        effective_args.moead_variation = merge_variation_overrides(
            getattr(args, "moead_variation", None), _override_mapping(override, "moead")
        )
        effective_args.smsemoa_variation = merge_variation_overrides(
            getattr(args, "smsemoa_variation", None), _override_mapping(override, "smsemoa")
        )
        effective_args.nsgaiii_variation = merge_variation_overrides(
            getattr(args, "nsgaiii_variation", None), _override_mapping(override, "nsgaiii")
        )
        effective_args.spea2_variation = merge_variation_overrides(
            getattr(args, "spea2_variation", None), _override_mapping(override, "spea2")
        )
        effective_args.ibea_variation = merge_variation_overrides(
            getattr(args, "ibea_variation", None), _override_mapping(override, "ibea")
        )
        effective_args.smpso_variation = merge_variation_overrides(
            getattr(args, "smpso_variation", None), _override_mapping(override, "smpso")
        )
        effective_args.agemoea_variation = merge_variation_overrides(
            getattr(args, "agemoea_variation", None), _override_mapping(override, "agemoea")
        )
        effective_args.rvea_variation = merge_variation_overrides(
            getattr(args, "rvea_variation", None), _override_mapping(override, "rvea")
        )
        effective_args.effective_problem_override = override

        (
            effective_args.nsgaii_variation,
            effective_args.moead_variation,
            effective_args.smsemoa_variation,
            effective_args.nsgaiii_variation,
            effective_args.spea2_variation,
            effective_args.ibea_variation,
            effective_args.smpso_variation,
            effective_args.agemoea_variation,
            effective_args.rvea_variation,
        ) = normalize_variations(
            nsgaii_variation=effective_args.nsgaii_variation,
            moead_variation=effective_args.moead_variation,
            smsemoa_variation=effective_args.smsemoa_variation,
            nsgaiii_variation=effective_args.nsgaiii_variation,
            spea2_variation=effective_args.spea2_variation,
            ibea_variation=effective_args.ibea_variation,
            smpso_variation=effective_args.smpso_variation,
            agemoea_variation=effective_args.agemoea_variation,
            rvea_variation=effective_args.rvea_variation,
        )

        if multiple:
            _logger().info("%s", "\n" + "#" * 80)
            _logger().info(
                "Problem %s/%s: %s (%s)",
                idx,
                len(selections),
                effective_selection.spec.label,
                effective_selection.spec.key,
            )
            _logger().info("%s", "#" * 80 + "\n")

        hv_stop_config = None
        if effective_args.hv_threshold is not None:
            hv_stop_config = build_hv_stop_config(
                effective_args.hv_threshold,
                effective_args.hv_reference_front,
                effective_selection.spec.key,
                n_obj=effective_selection.n_obj,
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
            agemoea_variation=effective_args.agemoea_variation,
            rvea_variation=effective_args.rvea_variation,
            include_external=effective_args.include_external,
            config_source=config_source,
            config_spec=config_spec,
            problem_override=override,
            track_genealogy=effective_args.track_genealogy,
            autodiff_constraints=effective_args.autodiff_constraints,
            live_viz_factory=live_viz_factory,
            plotter=plotter,
        )


__all__ = ["run_from_args"]
