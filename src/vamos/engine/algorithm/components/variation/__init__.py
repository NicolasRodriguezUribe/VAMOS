# algorithm/components/variation/__init__.py
"""
Variation operators and pipelines.

This subpackage contains:
- pipeline: VariationPipeline class for combining crossover + mutation
- helpers: Low-level variation utilities and operator dispatch
- core: High-level variation API and parameter preparation
"""

from vamos.engine.algorithm.components.variation.core import (
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
