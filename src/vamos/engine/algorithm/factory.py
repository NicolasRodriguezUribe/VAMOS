from __future__ import annotations

from collections.abc import Callable, Mapping

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
from vamos.engine.algorithm.config import GenericAlgorithmConfig
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.engine.algorithm.registry import resolve_algorithm
from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.config.variation import VariationConfig
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.types import ProblemProtocol


def _merge_variation_aliases(
    variations: Mapping[str, VariationConfig | None] | None,
    *,
    nsgaii_variation: VariationConfig | None = None,
    moead_variation: VariationConfig | None = None,
    smsemoa_variation: VariationConfig | None = None,
    nsgaiii_variation: VariationConfig | None = None,
    spea2_variation: VariationConfig | None = None,
    ibea_variation: VariationConfig | None = None,
    smpso_variation: VariationConfig | None = None,
    agemoea_variation: VariationConfig | None = None,
    rvea_variation: VariationConfig | None = None,
) -> dict[str, VariationConfig | None]:
    merged = dict(variations or {})
    aliases = {
        "nsgaii": nsgaii_variation,
        "moead": moead_variation,
        "smsemoa": smsemoa_variation,
        "nsgaiii": nsgaiii_variation,
        "spea2": spea2_variation,
        "ibea": ibea_variation,
        "smpso": smpso_variation,
        "agemoea": agemoea_variation,
        "rvea": rvea_variation,
    }
    for algorithm_name, override in aliases.items():
        if override is None:
            continue
        existing = merged.get(algorithm_name)
        if existing is not None and existing != override:
            raise TypeError(f"Conflicting variation overrides provided for '{algorithm_name}'.")
        merged[algorithm_name] = override
    return merged


def _plugin_default_config(
    problem: ProblemProtocol,
    config: ExperimentConfig,
) -> GenericAlgorithmConfig:
    return GenericAlgorithmConfig(
        {
            "pop_size": config.population_size,
            "offspring_size": config.offspring_size(),
            "n_var": problem.n_var,
            "n_obj": problem.n_obj,
            "encoding": normalize_encoding(getattr(problem, "encoding", "real")),
        }
    )


def build_algorithm(
    algorithm_name: str,
    engine_name: str,
    problem: ProblemProtocol,
    config: ExperimentConfig,
    *,
    algorithm_config: AlgorithmConfigProtocol | None = None,
    external_archive: ExternalArchiveConfig | None = None,
    selection_pressure: int = 2,
    variations: Mapping[str, VariationConfig | None] | None = None,
    track_genealogy: bool = False,
    nsgaii_variation: VariationConfig | None = None,
    moead_variation: VariationConfig | None = None,
    smsemoa_variation: VariationConfig | None = None,
    nsgaiii_variation: VariationConfig | None = None,
    spea2_variation: VariationConfig | None = None,
    ibea_variation: VariationConfig | None = None,
    smpso_variation: VariationConfig | None = None,
    agemoea_variation: VariationConfig | None = None,
    rvea_variation: VariationConfig | None = None,
) -> tuple[object, AlgorithmConfigProtocol]:
    """
    Factory to build the algorithm instance.
    """
    kernel = resolve_kernel(engine_name)
    algorithm_key = algorithm_name.lower()
    variations = _merge_variation_aliases(
        variations,
        nsgaii_variation=nsgaii_variation,
        moead_variation=moead_variation,
        smsemoa_variation=smsemoa_variation,
        nsgaiii_variation=nsgaiii_variation,
        spea2_variation=spea2_variation,
        ibea_variation=ibea_variation,
        smpso_variation=smpso_variation,
        agemoea_variation=agemoea_variation,
        rvea_variation=rvea_variation,
    )
    pop_size = config.population_size
    offspring_size = config.offspring_size()

    if algorithm_config is not None:
        algo_ctor = resolve_algorithm(algorithm_key)
        return algo_ctor(dict(algorithm_config.to_dict()), kernel), algorithm_config

    builders: dict[str, Callable[[], tuple[object, AlgorithmConfigProtocol]]] = {
        "nsgaii": lambda: build_nsgaii_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            offspring_size=offspring_size,
            selection_pressure=selection_pressure,
            external_archive=external_archive,
            nsgaii_variation=variations.get("nsgaii"),
            track_genealogy=track_genealogy,
        ),
        "moead": lambda: build_moead_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            moead_variation=variations.get("moead"),
        ),
        "smsemoa": lambda: build_smsemoa_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            smsemoa_variation=variations.get("smsemoa"),
        ),
        "nsgaiii": lambda: build_nsgaiii_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            nsgaiii_variation=variations.get("nsgaiii"),
            selection_pressure=selection_pressure,
        ),
        "spea2": lambda: build_spea2_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            selection_pressure=selection_pressure,
            external_archive=external_archive,
            spea2_variation=variations.get("spea2"),
        ),
        "ibea": lambda: build_ibea_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            selection_pressure=selection_pressure,
            ibea_variation=variations.get("ibea"),
        ),
        "smpso": lambda: build_smpso_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            external_archive=external_archive,
            smpso_variation=variations.get("smpso"),
        ),
        "agemoea": lambda: build_agemoea_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            agemoea_variation=variations.get("agemoea"),
        ),
        "rvea": lambda: build_rvea_algorithm(
            kernel=kernel,
            problem=problem,
            pop_size=pop_size,
            rvea_variation=variations.get("rvea"),
        ),
    }

    builder = builders.get(algorithm_key)
    if builder is not None:
        return builder()

    plugin_config = _plugin_default_config(problem, config)
    algo_ctor = resolve_algorithm(algorithm_key)
    return algo_ctor(dict(plugin_config.to_dict()), kernel), plugin_config
