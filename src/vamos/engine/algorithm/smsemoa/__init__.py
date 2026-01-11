"""
SMSEMOA algorithm module.

This package provides the SMS-EMOA (S-Metric Selection EMOA)
implementation with modular components:
- `core.py`: main SMSEMOA class (run/ask/tell loop)
- `setup.py`: initialization/config helpers
- `state.py`: SMSEMOAState + result building
- `operators/policies/smsemoa.py`: variation operator building
- `helpers.py`: reference point management, survival selection

References:
    N. Beume, B. Naujoks, and M. Emmerich, "SMS-EMOA: Multiobjective Selection
    Based on Dominated Hypervolume," European Journal of Operational Research,
    vol. 181, no. 3, 2007.
"""

from .smsemoa import SMSEMOA
from .helpers import (
    evaluate_population_with_constraints,
    initialize_reference_point,
    survival_selection,
    update_reference_point,
)
from vamos.operators.policies.smsemoa import (
    BINARY_CROSSOVER,
    BINARY_MUTATION,
    INT_CROSSOVER,
    INT_MUTATION,
    build_variation_operators,
)
from .initialization import initialize_population, initialize_smsemoa_run
from .state import SMSEMOAState, build_smsemoa_result

__all__ = [
    "SMSEMOA",
    # Helpers
    "evaluate_population_with_constraints",
    "initialize_reference_point",
    "survival_selection",
    "update_reference_point",
    # Operators
    "BINARY_CROSSOVER",
    "BINARY_MUTATION",
    "INT_CROSSOVER",
    "INT_MUTATION",
    "build_variation_operators",
    # Setup
    "initialize_population",
    "initialize_smsemoa_run",
    # State
    "SMSEMOAState",
    "build_smsemoa_result",
]
