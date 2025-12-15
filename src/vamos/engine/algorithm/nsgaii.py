"""
NSGA-II algorithm module.

This file mirrors the other algorithms in this package (e.g. `moead.py`, `nsga3.py`)
while keeping the implementation split across focused modules:
- `nsgaii_core.py`: main NSGAII class (run/ask/tell loop)
- `nsgaii_setup.py`: initialization/config helpers
- `nsgaii_state.py`: NSGAIIState + result/genealogy helpers
- `nsgaii_operators.py`: operator pool + adaptive selection wiring
- `nsgaii_helpers.py`: mating pool + survival helpers
"""

from .nsgaii_core import NSGAII
from .nsgaii_helpers import (
    build_mating_pool,
    feasible_nsga2_survival,
    match_ids,
    operator_success_stats,
    generation_contributions,
)

__all__ = [
    "NSGAII",
    "build_mating_pool",
    "feasible_nsga2_survival",
    "match_ids",
    "operator_success_stats",
    "generation_contributions",
]
