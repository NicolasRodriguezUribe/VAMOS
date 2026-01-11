"""
MOEA/D algorithm module.

This package provides the MOEA/D (Multi-Objective Evolutionary Algorithm based on
Decomposition) implementation with modular components:
- `core.py`: main MOEAD class (run/ask/tell loop)
- `setup.py`: initialization/config helpers
- `state.py`: MOEADState + result building
- `operators/policies/moead.py`: operator registries and building
- `helpers.py`: aggregation functions + neighborhood update

References:
    Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on
    Decomposition," IEEE Trans. Evolutionary Computation, vol. 11, no. 6, 2007.
"""

from .moead import MOEAD
from .helpers import (
    build_aggregator,
    compute_neighbors,
    modified_tchebycheff,
    pbi,
    tchebycheff,
    update_neighborhood,
    weighted_sum,
)
from vamos.operators.policies.moead import (
    BINARY_CROSSOVER,
    BINARY_MUTATION,
    INT_CROSSOVER,
    INT_MUTATION,
    build_variation_operators,
)
from .initialization import initialize_moead_run, initialize_population
from .state import MOEADState, build_moead_result

__all__ = [
    "MOEAD",
    # Helpers
    "build_aggregator",
    "compute_neighbors",
    "modified_tchebycheff",
    "pbi",
    "tchebycheff",
    "update_neighborhood",
    "weighted_sum",
    # Operators
    "BINARY_CROSSOVER",
    "BINARY_MUTATION",
    "INT_CROSSOVER",
    "INT_MUTATION",
    "build_variation_operators",
    # Setup
    "initialize_moead_run",
    "initialize_population",
    # State
    "MOEADState",
    "build_moead_result",
]
