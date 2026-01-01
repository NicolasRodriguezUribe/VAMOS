"""
High-level variation utilities.

This module provides a small, explicit re-export surface over the variation
subpackage (pipeline + helper registries).
"""

from __future__ import annotations

from vamos.engine.algorithm.components.variation.pipeline import VariationPipeline
from vamos.engine.algorithm.components.variation.helpers import (
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

__all__ = [
    "VariationPipeline",
    "prepare_mutation_params",
    "validate_operator_support",
    "resolve_prob_expression",
    "PERM_CROSSOVER",
    "PERM_MUTATION",
    "BINARY_CROSSOVER",
    "BINARY_MUTATION",
    "INT_CROSSOVER",
    "INT_MUTATION",
    "MIXED_CROSSOVER",
    "MIXED_MUTATION",
]
