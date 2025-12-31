"""
UX/analysis utilities; not required for core optimization runtime.
"""
from __future__ import annotations

from vamos.ux.analysis.core_objective_reduction import (
    ObjectiveReductionConfig,
    ObjectiveReducer,
    reduce_objectives,
)
from vamos.ux.analysis.mcdm import (
    MCDMResult,
    knee_point_scores,
    reference_point_scores,
    tchebycheff_scores,
    weighted_sum_scores,
)
from vamos.ux.analysis.stats import (
    FriedmanResult,
    WilcoxonResult,
    compute_ranks,
    friedman_test,
    pairwise_wilcoxon,
    plot_critical_distance,
)
from vamos.ux.visualization import (
    plot_hv_convergence,
    plot_parallel_coordinates,
    plot_pareto_front_2d,
    plot_pareto_front_3d,
)

__all__ = [
    # Objective reduction
    "ObjectiveReductionConfig",
    "ObjectiveReducer",
    "reduce_objectives",
    # MCDM
    "MCDMResult",
    "weighted_sum_scores",
    "tchebycheff_scores",
    "reference_point_scores",
    "knee_point_scores",
    # Stats
    "FriedmanResult",
    "WilcoxonResult",
    "compute_ranks",
    "friedman_test",
    "pairwise_wilcoxon",
    "plot_critical_distance",
    # Visualization helpers
    "plot_pareto_front_2d",
    "plot_pareto_front_3d",
    "plot_parallel_coordinates",
    "plot_hv_convergence",
]
