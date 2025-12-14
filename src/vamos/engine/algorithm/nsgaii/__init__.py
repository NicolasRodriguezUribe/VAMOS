# algorithm/nsgaii/__init__.py
"""
NSGA-II algorithm package.

This package contains the NSGA-II implementation split into focused modules:
- core: Main NSGAII class with run/ask/tell interface
- setup: Initialization and configuration helpers
- state: NSGAIIState dataclass for algorithm state management
- operators: Operator pool building and adaptive selection
- helpers: Mating pool, survival selection, and utility functions
"""
from vamos.engine.algorithm.nsgaii.core import NSGAII
from vamos.engine.algorithm.nsgaii.helpers import (
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
