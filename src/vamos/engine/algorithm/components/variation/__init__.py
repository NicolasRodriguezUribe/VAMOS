"""
Compatibility wrapper around :mod:`vamos.engine.variation`.
"""

from vamos.engine.variation import (
    VariationPipeline,
    prepare_mutation_params,
    BINARY_CROSSOVER,
    BINARY_MUTATION,
    INT_CROSSOVER,
    INT_MUTATION,
    MIXED_CROSSOVER,
    MIXED_MUTATION,
    PERM_CROSSOVER,
    PERM_MUTATION,
    resolve_prob_expression,
    validate_operator_support,
)

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
