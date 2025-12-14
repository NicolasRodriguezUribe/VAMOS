# algorithm/nsgaii.py
"""
Backward-compatibility shim for NSGA-II.

The implementation has moved to vamos.engine.algorithm.nsgaii/ package.
This module re-exports the public API for backward compatibility.
"""
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.engine.algorithm.nsgaii.helpers import (
    build_mating_pool,
    feasible_nsga2_survival,
    match_ids,
    operator_success_stats,
    generation_contributions,
)

# Legacy aliases (underscore-prefixed)
_build_mating_pool = build_mating_pool
_feasible_nsga2_survival = feasible_nsga2_survival
_match_ids = match_ids
_operator_success_stats = operator_success_stats
_generation_contributions = generation_contributions

__all__ = [
    "NSGAII",
    "build_mating_pool",
    "feasible_nsga2_survival",
    "match_ids",
    "operator_success_stats",
    "generation_contributions",
    "_build_mating_pool",
    "_feasible_nsga2_survival",
    "_match_ids",
    "_operator_success_stats",
    "_generation_contributions",
]
