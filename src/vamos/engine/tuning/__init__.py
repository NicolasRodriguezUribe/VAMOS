"""Auto-tuning utilities for VAMOS.

Parameter types use signature: ParamType(name, ...)
- Real(name, low, high, log=False)
- Int(name, low, high, log=False)
- Categorical(name, choices)
- Boolean(name)

Also provides RandomSearchTuner, RacingTuner, AlgorithmConfigSpace,
and config space builders for each algorithm.
"""

from .racing import (
    # Parameter types
    ParamSpace,
    Real,
    Int,
    Categorical,
    Boolean,
    Condition,
    ConditionalBlock,
    ParamType,
    BaseParam,
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
    # Parameter types
    "ParamSpace",
    "Real",
    "Int",
    "Categorical",
    "Boolean",
    "Condition",
    "ConditionalBlock",
    "ParamType",
    "BaseParam",
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
