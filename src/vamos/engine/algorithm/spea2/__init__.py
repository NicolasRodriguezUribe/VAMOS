"""
SPEA2 algorithm module.

This package provides the SPEA2 (Strength Pareto Evolutionary Algorithm 2)
implementation with modular components:
- `core.py`: main SPEA2 class (run/ask/tell loop)
- `setup.py`: initialization/config helpers
- `state.py`: SPEA2State + result building
- `operators.py`: variation operator building
- `helpers.py`: environmental selection, fitness calculation, truncation

References:
    E. Zitzler, M. Laumanns, and L. Thiele, "SPEA2: Improving the Strength
    Pareto Evolutionary Algorithm," TIK-Report 103, ETH Zurich, 2001.
"""

from .spea2 import SPEA2
from .helpers import (
    compute_selection_metrics,
    dominance_matrix,
    environmental_selection,
    spea2_fitness,
    truncate_by_distance,
)
from .operators import build_variation_operators
from .initialization import initialize_spea2_run
from .state import SPEA2State, build_spea2_result

__all__ = [
    "SPEA2",
    # Helpers
    "compute_selection_metrics",
    "dominance_matrix",
    "environmental_selection",
    "spea2_fitness",
    "truncate_by_distance",
    # Operators
    "build_variation_operators",
    # Setup
    "initialize_spea2_run",
    # State
    "SPEA2State",
    "build_spea2_result",
]
