"""
NSGA-III algorithm module.

This package provides the NSGA-III (Non-dominated Sorting Genetic Algorithm III)
implementation with modular components:
- `core.py`: main NSGAIII class (run/ask/tell loop)
- `setup.py`: initialization/config helpers
- `state.py`: NSGAIIIState + result building
- `operators/policies/nsgaiii.py`: variation operator building
- `helpers.py`: reference point niching, survival selection

References:
    K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm
    Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving
    Problems With Box Constraints," IEEE Trans. Evolutionary Computation,
    vol. 18, no. 4, 2014.
"""

from .nsgaiii import NSGAIII
from .helpers import associate, evaluate_population_with_constraints, nsgaiii_survival
from vamos.operators.policies.nsgaiii import build_variation_operators
from .initialization import initialize_nsgaiii_run
from .state import NSGAIIIState, build_nsgaiii_result

__all__ = [
    "NSGAIII",
    # Helpers
    "associate",
    "evaluate_population_with_constraints",
    "nsgaiii_survival",
    # Operators
    "build_variation_operators",
    # Setup
    "initialize_nsgaiii_run",
    # State
    "NSGAIIIState",
    "build_nsgaiii_result",
]
