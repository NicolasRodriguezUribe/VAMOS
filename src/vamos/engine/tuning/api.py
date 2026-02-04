from __future__ import annotations

"""
Facade for tuning utilities (racing-style tuning).

Parameter types use signature: ParamType(name, ...)
- Real(name, low, high, log=False)
- Int(name, low, high, log=False)
- Categorical(name, choices)
- Boolean(name)

For full control, import from `vamos.engine.tuning.*`.
"""

from vamos.engine.tuning import (
    # Ablation planning
    AblationPlan,
    AblationTask,
    AblationVariant,
    build_ablation_plan,
    # Tuners
    AlgorithmConfigSpace,
    RandomSearchTuner,
    RacingTuner,
    # Parameter types
    ParamSpace,
    Real,
    Int,
    Categorical,
    Boolean,
    Condition,
    ConditionalBlock,
    # Task/scenario
    TuningTask,
    EvalContext,
    Instance,
    Scenario,
    # Config space builders
    build_ibea_config_space,
    build_ibea_binary_config_space,
    build_ibea_integer_config_space,
    build_moead_config_space,
    build_moead_binary_config_space,
    build_moead_integer_config_space,
    build_moead_permutation_config_space,
    build_nsgaiii_config_space,
    build_nsgaiii_binary_config_space,
    build_nsgaiii_integer_config_space,
    build_nsgaii_config_space,
    build_nsgaii_binary_config_space,
    build_nsgaii_integer_config_space,
    build_nsgaii_permutation_config_space,
    build_nsgaii_mixed_config_space,
    build_smpso_config_space,
    build_agemoea_config_space,
    build_rvea_config_space,
    build_smsemoa_config_space,
    build_smsemoa_binary_config_space,
    build_smsemoa_integer_config_space,
    build_spea2_config_space,
    config_from_assignment,
)

__all__ = [
    # Ablation planning
    "AblationPlan",
    "AblationTask",
    "AblationVariant",
    "build_ablation_plan",
    # Tuners
    "AlgorithmConfigSpace",
    "RandomSearchTuner",
    "RacingTuner",
    # Parameter types
    "ParamSpace",
    "Real",
    "Int",
    "Categorical",
    "Boolean",
    "Condition",
    "ConditionalBlock",
    # Task/scenario
    "TuningTask",
    "EvalContext",
    "Instance",
    "Scenario",
    # Config space builders
    "build_nsgaii_config_space",
    "build_nsgaii_permutation_config_space",
    "build_nsgaii_mixed_config_space",
    "build_nsgaii_binary_config_space",
    "build_nsgaii_integer_config_space",
    "build_nsgaiii_config_space",
    "build_nsgaiii_binary_config_space",
    "build_nsgaiii_integer_config_space",
    "build_moead_config_space",
    "build_moead_permutation_config_space",
    "build_moead_binary_config_space",
    "build_moead_integer_config_space",
    "build_smsemoa_config_space",
    "build_smsemoa_binary_config_space",
    "build_smsemoa_integer_config_space",
    "build_spea2_config_space",
    "build_ibea_config_space",
    "build_ibea_binary_config_space",
    "build_ibea_integer_config_space",
    "build_smpso_config_space",
    "build_agemoea_config_space",
    "build_rvea_config_space",
    "config_from_assignment",
]
