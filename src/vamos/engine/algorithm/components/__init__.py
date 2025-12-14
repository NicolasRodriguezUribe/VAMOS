# algorithm/components/__init__.py
"""
Shared algorithm components.

This package contains building blocks used across multiple algorithms:
- archive: External archive implementations (crowding, hypervolume)
- population: Population initialization and evaluation
- selection: Parent selection strategies
- termination: Termination criteria and trackers
- hypervolume: Hypervolume calculation utilities
- weight_vectors: Weight vector generation for decomposition
- variation: Variation operators and pipelines (subpackage)
"""
from vamos.engine.algorithm.components.archive import (
    CrowdingDistanceArchive,
    HypervolumeArchive,
    _single_front_crowding,
)
from vamos.engine.algorithm.components.hypervolume import hypervolume
from vamos.engine.algorithm.components.population import (
    evaluate_population_with_constraints,
    initialize_population,
    resolve_bounds,
)
from vamos.engine.algorithm.components.selection import TournamentSelection, RandomSelection
from vamos.engine.algorithm.components.termination import HVTracker

__all__ = [
    # archive
    "CrowdingDistanceArchive",
    "HypervolumeArchive",
    "_single_front_crowding",
    # hypervolume
    "hypervolume",
    # population
    "evaluate_population_with_constraints",
    "initialize_population",
    "resolve_bounds",
    # selection
    "TournamentSelection",
    "RandomSelection",
    # termination
    "HVTracker",
]
