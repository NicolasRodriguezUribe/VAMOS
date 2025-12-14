# algorithm/variation.py
"""
Backward-compatibility shim for variation module.

The implementation has moved to vamos.engine.algorithm.components.variation.
This module re-exports the public API for backward compatibility.
"""
from vamos.engine.algorithm.components.variation import (
    VariationPipeline,
    prepare_mutation_params,
)
from vamos.engine.algorithm.components.variation.helpers import (
    resolve_prob_expression,
    validate_operator_support,
    PERM_CROSSOVER,
    PERM_MUTATION,
    BINARY_CROSSOVER,
    BINARY_MUTATION,
    INT_CROSSOVER,
    INT_MUTATION,
    MIXED_CROSSOVER,
    MIXED_MUTATION,
)

# Legacy aliases
_PERM_CROSSOVER = PERM_CROSSOVER
_PERM_MUTATION = PERM_MUTATION
_BINARY_CROSSOVER = BINARY_CROSSOVER
_BINARY_MUTATION = BINARY_MUTATION
_INT_CROSSOVER = INT_CROSSOVER
_INT_MUTATION = INT_MUTATION
_MIXED_CROSSOVER = MIXED_CROSSOVER
_MIXED_MUTATION = MIXED_MUTATION

__all__ = [
    "VariationPipeline",
    "prepare_mutation_params",
    "resolve_prob_expression",
    "validate_operator_support",
    "PERM_CROSSOVER",
    "PERM_MUTATION",
    "BINARY_CROSSOVER",
    "BINARY_MUTATION",
    "INT_CROSSOVER",
    "INT_MUTATION",
    "MIXED_CROSSOVER",
    "MIXED_MUTATION",
]
