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
from vamos.algorithm.components.archive import (
    CrowdingDistanceArchive,
    HypervolumeArchive,
    _single_front_crowding,
)
from vamos.algorithm.components.hypervolume import hypervolume
from vamos.algorithm.components.population import (
    evaluate_population_with_constraints,
    initialize_population,
    resolve_bounds,
)
from vamos.algorithm.components.selection import TournamentSelection, RandomSelection
from vamos.algorithm.components.termination import HVTracker

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
