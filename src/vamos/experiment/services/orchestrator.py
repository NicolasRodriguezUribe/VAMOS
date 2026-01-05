from __future__ import annotations

from typing import Any

from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import ProblemSelection
from vamos.hooks import LiveVisualization
from vamos.experiment.execution import run_single as execute_run_single

from vamos.experiment.services.config import VariationConfig, normalize_variations
from vamos.experiment.services.factory import build_algorithm_from_spec


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
