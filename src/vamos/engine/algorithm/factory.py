from __future__ import annotations

from vamos.archive import ExternalArchiveConfig
from vamos.engine.algorithm.builders import (
    build_agemoea_algorithm,
    build_ibea_algorithm,
    build_moead_algorithm,
    build_nsgaii_algorithm,
    build_nsgaiii_algorithm,
    build_rvea_algorithm,
    build_smpso_algorithm,
    build_smsemoa_algorithm,
    build_spea2_algorithm,
)
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.engine.config.variation import VariationConfig
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.types import ProblemProtocol


def build_algorithm(
    algorithm_name: str,
    engine_name: str,
    problem: ProblemProtocol,
    config: ExperimentConfig,
    *,
    external_archive: ExternalArchiveConfig | None = None,
    selection_pressure: int = 2,
    nsgaii_variation: VariationConfig | None = None,
    moead_variation: VariationConfig | None = None,
    smsemoa_variation: VariationConfig | None = None,
    nsgaiii_variation: VariationConfig | None = None,
    spea2_variation: VariationConfig | None = None,
    ibea_variation: VariationConfig | None = None,
    smpso_variation: VariationConfig | None = None,
    agemoea_variation: VariationConfig | None = None,
    rvea_variation: VariationConfig | None = None,
    track_genealogy: bool = False,
) -> tuple[object, AlgorithmConfigProtocol]:
    """
    Factory to build the algorithm instance.
    """
    kernel = resolve_kernel(engine_name)
    pop_size = config.population_size
    offspring_size = config.offspring_size()
    # Note: seed is available via config.seed but algorithms handle their own RNG

    if algorithm_name == "nsgaii":
        return build_nsgaii_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            offspring_size=offspring_size,
            selection_pressure=selection_pressure,
            external_archive=external_archive,
            nsgaii_variation=nsgaii_variation,
            track_genealogy=track_genealogy,
        )

    elif algorithm_name == "moead":
        return build_moead_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            moead_variation=moead_variation,
        )

    elif algorithm_name == "smsemoa":
        return build_smsemoa_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            smsemoa_variation=smsemoa_variation,
        )

    elif algorithm_name == "nsgaiii":
        return build_nsgaiii_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            nsgaiii_variation=nsgaiii_variation,
            selection_pressure=selection_pressure,
        )

    elif algorithm_name == "spea2":
        return build_spea2_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            selection_pressure=selection_pressure,
            external_archive=external_archive,
            spea2_variation=spea2_variation,
        )

    elif algorithm_name == "ibea":
        return build_ibea_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            selection_pressure=selection_pressure,
            ibea_variation=ibea_variation,
        )

    elif algorithm_name == "smpso":
        return build_smpso_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            external_archive=external_archive,
            smpso_variation=smpso_variation,
        )

    elif algorithm_name == "agemoea":
        return build_agemoea_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            agemoea_variation=agemoea_variation,  # Using generic **kwargs if not in signature? No, need to add to signature.
        )

    elif algorithm_name == "rvea":
        return build_rvea_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            rvea_variation=rvea_variation,
        )

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")
