"""
NSGA-II algorithm module.

This package provides the NSGA-II (Non-dominated Sorting Genetic Algorithm II)
implementation with modular components:
- `core.py`: main NSGAII class (run/ask/tell loop)
- `setup.py`: initialization/config helpers
- `state.py`: NSGAIIState + result/genealogy helpers
- `operators/policies/nsgaii.py`: operator pool + adaptive selection wiring
- `helpers.py`: mating pool + survival helpers

Example:
    >>> from vamos.engine.algorithm.nsgaii import NSGAII
    >>> from vamos.engine.algorithm.config import NSGAIIConfig
    >>> config = NSGAIIConfig().pop_size(100).crossover("sbx", prob=0.9).fixed()
    >>> algo = NSGAII(config.to_dict(), kernel)
    >>> result = algo.run(problem, ("n_eval", 10000), seed=42)
"""

from .nsgaii import NSGAII
from .helpers import (
    build_mating_pool,
    feasible_nsga2_survival,
    generation_contributions,
    match_ids,
    operator_success_stats,
)

__all__ = [
    "NSGAII",
    "build_mating_pool",
    "feasible_nsga2_survival",
    "match_ids",
    "operator_success_stats",
    "generation_contributions",
]
