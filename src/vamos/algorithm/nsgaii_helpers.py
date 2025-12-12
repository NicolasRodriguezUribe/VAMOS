# algorithm/nsgaii_helpers.py
"""
Backward-compatibility shim for nsgaii_helpers module.

The implementation has moved to vamos.algorithm.nsgaii.helpers.
This module re-exports the public API for backward compatibility.
"""
from vamos.algorithm.nsgaii.helpers import (
    build_mating_pool,
    feasible_nsga2_survival,
    match_ids,
    operator_success_stats,
    generation_contributions,
)

__all__ = [
    "build_mating_pool",
    "feasible_nsga2_survival",
    "match_ids",
    "operator_success_stats",
    "generation_contributions",
]
