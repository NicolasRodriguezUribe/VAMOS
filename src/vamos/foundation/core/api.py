"""
Public-facing API surface for VAMOS.

This module re-exports stable entrypoints intended for library consumers.
Internal/experimental modules (runner, CLI helpers, tuning pipelines, etc.)
should be imported explicitly from their modules instead of via the package root.
"""
from __future__ import annotations

from vamos.engine.algorithm.config import (
    IBEAConfig,
    IBEAConfigData,
    MOEADConfig,
    MOEADConfigData,
    NSGAIIConfig,
    NSGAIIConfigData,
    NSGAIIIConfig,
    NSGAIIIConfigData,
    SMPSOConfig,
    SMPSOConfigData,
    SMSEMOAConfig,
    SMSEMOAConfigData,
    SPEA2Config,
    SPEA2ConfigData,
)
from vamos.experiment.diagnostics.self_check import run_self_check
from vamos.foundation.constraints import (
    ConstraintHandlingStrategy,
    ConstraintInfo,
    CVAsObjectiveStrategy,
    EpsilonConstraintStrategy,
    FeasibilityFirstStrategy,
    PenaltyCVStrategy,
    compute_constraint_info,
    get_constraint_strategy,
)
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.core.optimize import OptimizeConfig, OptimizationResult, optimize
from vamos.foundation.problem.registry import (
    ProblemSelection,
    ProblemSpec,
    available_problem_names,
    make_problem_selection,
)
from vamos.ux.analysis.core_objective_reduction import ObjectiveReductionConfig, ObjectiveReducer, reduce_objectives
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
    # Optimization facade
    "optimize",
    "OptimizeConfig",
    "OptimizationResult",
    # Experiment config
    "ExperimentConfig",
    # Algorithm configs
    "NSGAIIConfig",
    "NSGAIIConfigData",
    "MOEADConfig",
    "MOEADConfigData",
    "SMSEMOAConfig",
    "SMSEMOAConfigData",
    "NSGAIIIConfig",
    "NSGAIIIConfigData",
    "SPEA2Config",
    "SPEA2ConfigData",
    "IBEAConfig",
    "IBEAConfigData",
    "SMPSOConfig",
    "SMPSOConfigData",
    # Problem registry
    "ProblemSpec",
    "ProblemSelection",
    "available_problem_names",
    "make_problem_selection",
    # Objective reduction
    "ObjectiveReductionConfig",
    "ObjectiveReducer",
    "reduce_objectives",
    # Constraints
    "ConstraintInfo",
    "ConstraintHandlingStrategy",
    "FeasibilityFirstStrategy",
    "PenaltyCVStrategy",
    "CVAsObjectiveStrategy",
    "EpsilonConstraintStrategy",
    "compute_constraint_info",
    "get_constraint_strategy",
    # Visualization helpers
    "plot_pareto_front_2d",
    "plot_pareto_front_3d",
    "plot_parallel_coordinates",
    "plot_hv_convergence",
    # Self-check
    "run_self_check",
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
]
