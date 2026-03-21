"""Compatibility facade for MOEA/D aggregation and neighborhood helpers."""

from __future__ import annotations

from .aggregation import (
    ZERO_WEIGHT_EPS,
    build_aggregator,
    modified_tchebycheff,
    pbi,
    resolve_aggregation_spec,
    tchebycheff,
    weighted_sum,
)
from .neighborhood import compute_neighbors, update_neighborhood, update_neighborhood_batch

__all__ = [
    "ZERO_WEIGHT_EPS",
    "tchebycheff",
    "weighted_sum",
    "pbi",
    "modified_tchebycheff",
    "build_aggregator",
    "resolve_aggregation_spec",
    "compute_neighbors",
    "update_neighborhood",
    "update_neighborhood_batch",
]
