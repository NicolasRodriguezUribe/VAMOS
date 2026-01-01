"""
Statistical testing helpers for experiment comparisons.

Sourced from `vamos.ux.analysis.stats`.
"""

from __future__ import annotations

from vamos.ux.analysis.stats import (
    FriedmanResult,
    WilcoxonResult,
    compute_ranks,
    friedman_test,
    pairwise_wilcoxon,
    plot_critical_distance,
)

__all__ = [
    "FriedmanResult",
    "WilcoxonResult",
    "compute_ranks",
    "friedman_test",
    "pairwise_wilcoxon",
    "plot_critical_distance",
]
