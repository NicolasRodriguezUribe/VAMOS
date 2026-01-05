from __future__ import annotations

from typing import Any

from vamos.engine.algorithm.factory import build_algorithm
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.experiment.services.config import VariationConfig


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
