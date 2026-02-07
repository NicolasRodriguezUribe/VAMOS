"""
NSGA-II algorithm module.

This package provides the NSGA-II (Non-dominated Sorting Genetic Algorithm II)
implementation with modular components:
- `nsgaii.py`: main NSGAII class (thin orchestrator)
- `run.py`: run loop + checkpoints + live-viz notifications
- `setup.py`: initialization/config helpers
- `ask_tell.py`: ask/tell operations
- `state.py`: NSGAIIState + result/genealogy helpers
- `operators/policies/nsgaii.py`: operator pool + adaptive selection wiring
- `helpers.py`: mating pool + survival helpers

Example:
    >>> from vamos.engine.algorithm.nsgaii import NSGAII
    >>> from vamos.engine.algorithm.config import NSGAIIConfig
    >>> config = NSGAIIConfig.builder().pop_size(100).crossover("sbx", prob=0.9).build()
    >>> algo = NSGAII(config.to_dict(), kernel)
    >>> result = algo.run(problem, ("max_evaluations", 10000), seed=42)
"""

from .nsgaii import NSGAII
from .helpers import (
    build_mating_pool,
    feasible_nsga2_survival,
    generation_contributions,
    match_ids,
    operator_success_stats,
)
from .injection import ImmigrationManager, ImmigrantCandidate, ImmigrationStats

__all__ = [
    "NSGAII",
    "build_mating_pool",
    "feasible_nsga2_survival",
    "match_ids",
    "operator_success_stats",
    "generation_contributions",
    "ImmigrationManager",
    "ImmigrantCandidate",
    "ImmigrationStats",
]
