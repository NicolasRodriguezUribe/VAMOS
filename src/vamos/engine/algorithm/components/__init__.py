# algorithm/components/__init__.py
"""
Shared algorithm components.

This package contains building blocks used across multiple algorithms:
- archive: External archive implementations (crowding, hypervolume)
- base: Base algorithm infrastructure (state, setup, result building)
- population: Population initialization and evaluation
- selection: Parent selection strategies
- termination: Termination criteria and trackers
- weight_vectors: Weight vector generation for decomposition
- variation: Variation operators and pipelines (subpackage)
- protocol: Algorithm interface definitions and enums
- utils: Shared utility functions
"""

from vamos.engine.algorithm.components.archive import (
    CrowdingDistanceArchive,
    HypervolumeArchive,
    _single_front_crowding,
)
from vamos.engine.algorithm.components.archives import (
    resolve_archive_size,
    setup_archive,
    update_archive,
)
from vamos.engine.algorithm.components.hooks import (
    get_live_viz,
    notify_generation,
)
from vamos.engine.algorithm.components.lifecycle import (
    get_eval_strategy,
    setup_initial_population,
)
from vamos.engine.algorithm.components.metrics import setup_hv_tracker
from vamos.engine.algorithm.components.results import build_result
from vamos.engine.algorithm.components.state import AlgorithmState
from vamos.engine.algorithm.components.termination import parse_termination
from vamos.engine.algorithm.components.population import (
    initialize_population,
    resolve_bounds,
)
from vamos.engine.algorithm.components.protocol import (
    AlgorithmProtocol,
    InteractiveAlgorithmProtocol,
    SelectionMethod,
    SurvivalMethod,
    ConstraintMode,
)
from vamos.engine.algorithm.components.selection import TournamentSelection, RandomSelection
from vamos.engine.algorithm.components.termination import HVTracker
from vamos.engine.algorithm.components.utils import (
    resolve_prob_expression,
    resolve_bounds_array,
    validate_termination,
    parse_operator_config,
    compute_ideal_nadir,
    normalize_objectives,
)

__all__ = [
    # protocol
    "AlgorithmProtocol",
    "InteractiveAlgorithmProtocol",
    "SelectionMethod",
    "SurvivalMethod",
    "ConstraintMode",
    # base
    "AlgorithmState",
    "parse_termination",
    "setup_initial_population",
    "setup_archive",
    "update_archive",
    "resolve_archive_size",
    "setup_hv_tracker",
    "get_live_viz",
    "notify_generation",
    "get_eval_strategy",
    "build_result",
    # archive
    "CrowdingDistanceArchive",
    "HypervolumeArchive",
    "_single_front_crowding",
    # population
    "initialize_population",
    "resolve_bounds",
    # selection
    "TournamentSelection",
    "RandomSelection",
    # termination
    "HVTracker",
    # utils
    "resolve_prob_expression",
    "resolve_bounds_array",
    "validate_termination",
    "parse_operator_config",
    "compute_ideal_nadir",
    "normalize_objectives",
]
