"""
Backwards-compatible facade for variation helpers/pipeline.
"""
from __future__ import annotations

from vamos.algorithm.variation_pipeline import VariationPipeline
from vamos.algorithm.variation_helpers import (
    resolve_prob_expression,
    prepare_mutation_params,
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

# Aliases to ease imports that referenced module-level registries
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
    "validate_operator_support",
    "resolve_prob_expression",
    "_PERM_CROSSOVER",
    "_PERM_MUTATION",
    "_BINARY_CROSSOVER",
    "_BINARY_MUTATION",
    "_INT_CROSSOVER",
    "_INT_MUTATION",
    "_MIXED_CROSSOVER",
    "_MIXED_MUTATION",
]
