"""Auto-tuning utilities for VAMOS.

This package contains racing-style tuning utilities:
- ParamSpace, RandomSearchTuner, RacingTuner, etc.
- Config space builders for each algorithm
"""

from .racing import (
    # Parameter space
    ParamSpace,
    Real,
    Int,
    Categorical,
    Condition,
    # Samplers
    Sampler,
    UniformSampler,
    ModelBasedSampler,
    # Tuning task
    TuningTask,
    EvalContext,
    Instance,
    # Scenario
    Scenario,
    # Config space
    AlgorithmConfigSpace,
    BaseParam,
    BooleanParam,
    CategoricalIntegerParam,
    CategoricalParam,
    ConditionalBlock,
    FloatParam,
    IntegerParam,
    # Tuners
    RandomSearchTuner,
    RacingTuner,
    TrialResult,
    ConfigState,
    EliteEntry,
    # I/O
    filter_active_config,
    history_to_dict,
    save_history_json,
    save_history_csv,
    # Config space builders
    build_spea2_config_space,
    build_ibea_config_space,
    build_smpso_config_space,
    build_moead_config_space,
    build_nsgaiii_config_space,
    build_smsemoa_config_space,
    build_nsgaii_config_space,
    config_from_assignment,
)


__all__ = [
    # Parameter space
    "ParamSpace",
    "Real",
    "Int",
    "Categorical",
    "Condition",
    # Samplers
    "Sampler",
    "UniformSampler",
    "ModelBasedSampler",
    # Tuning task
    "TuningTask",
    "EvalContext",
    "Instance",
    # Scenario
    "Scenario",
    # Config space
    "AlgorithmConfigSpace",
    "BaseParam",
    "BooleanParam",
    "CategoricalIntegerParam",
    "CategoricalParam",
    "ConditionalBlock",
    "FloatParam",
    "IntegerParam",
    # Tuners
    "RandomSearchTuner",
    "RacingTuner",
    "TrialResult",
    "ConfigState",
    "EliteEntry",
    # I/O
    "filter_active_config",
    "history_to_dict",
    "save_history_json",
    "save_history_csv",
    # Config space builders
    "build_spea2_config_space",
    "build_ibea_config_space",
    "build_smpso_config_space",
    "build_moead_config_space",
    "build_nsgaiii_config_space",
    "build_smsemoa_config_space",
    "build_nsgaii_config_space",
    "config_from_assignment",
]
