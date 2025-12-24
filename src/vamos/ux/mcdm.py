"""
Multi-criteria decision-making helpers for post-processing fronts.
"""
from __future__ import annotations

from vamos.ux.analysis.mcdm import (
    MCDMResult,
    knee_point_scores,
    reference_point_scores,
    tchebycheff_scores,
    weighted_sum_scores,
)

__all__ = [
    "MCDMResult",
    "weighted_sum_scores",
    "tchebycheff_scores",
    "reference_point_scores",
    "knee_point_scores",
]

