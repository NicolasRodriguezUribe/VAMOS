# algorithm/population.py
"""
Backward-compatibility shim for population module.

The implementation has moved to vamos.engine.algorithm.components.population.
This module re-exports the public API for backward compatibility.
"""
from vamos.engine.algorithm.components.population import (
    evaluate_population,
    evaluate_population_with_constraints,
    initialize_population,
    resolve_bounds,
)

__all__ = [
    "evaluate_population",
    "evaluate_population_with_constraints",
    "initialize_population",
    "resolve_bounds",
]
