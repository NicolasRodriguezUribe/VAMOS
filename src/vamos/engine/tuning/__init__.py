"""Auto-tuning utilities for VAMOS.

Parameter types use signature: ParamType(name, ...)
- Real(name, low, high, log=False)
- Int(name, low, high, log=False)
- Categorical(name, choices)
- Boolean(name)

Also provides RandomSearchTuner, RacingTuner, AlgorithmConfigSpace,
and config space builders for each algorithm.
"""

from .ablation import AblationPlan, AblationTask, AblationVariant, build_ablation_plan
from .backends import ModelBasedTuner, available_model_based_backends
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
    # Two-phase tuner
    TwoPhaseTuner,
    TwoPhaseScenario,
    # I/O
    filter_active_config,
    history_to_dict,
    save_history_json,
    save_history_csv,
    save_checkpoint,
    load_checkpoint,
    # Config space builders
    build_spea2_config_space,
    build_ibea_config_space,
    build_ibea_binary_config_space,
    build_ibea_integer_config_space,
    build_smpso_config_space,
    build_agemoea_config_space,
    build_rvea_config_space,
    build_moead_config_space,
    build_moead_binary_config_space,
    build_moead_integer_config_space,
    build_moead_permutation_config_space,
    build_nsgaii_binary_config_space,
    build_nsgaiii_config_space,
    build_nsgaiii_binary_config_space,
    build_nsgaiii_integer_config_space,
    build_smsemoa_config_space,
    build_smsemoa_binary_config_space,
    build_smsemoa_integer_config_space,
    build_nsgaii_config_space,
    build_nsgaii_integer_config_space,
    build_nsgaii_permutation_config_space,
    build_nsgaii_mixed_config_space,
    config_from_assignment,
)


__all__ = [
    # Ablation planning
    "AblationPlan",
    "AblationTask",
    "AblationVariant",
    "build_ablation_plan",
    # External backends
    "ModelBasedTuner",
    "available_model_based_backends",
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
    "TwoPhaseTuner",
    "TwoPhaseScenario",
    # I/O
    "filter_active_config",
    "history_to_dict",
    "save_history_json",
    "save_history_csv",
    "save_checkpoint",
    "load_checkpoint",
    # Config space builders
    "build_spea2_config_space",
    "build_ibea_config_space",
    "build_ibea_binary_config_space",
    "build_ibea_integer_config_space",
    "build_smpso_config_space",
    "build_agemoea_config_space",
    "build_rvea_config_space",
    "build_moead_config_space",
    "build_moead_binary_config_space",
    "build_moead_integer_config_space",
    "build_moead_permutation_config_space",
    "build_nsgaiii_config_space",
    "build_nsgaiii_binary_config_space",
    "build_nsgaiii_integer_config_space",
    "build_smsemoa_config_space",
    "build_smsemoa_binary_config_space",
    "build_smsemoa_integer_config_space",
    "build_nsgaii_config_space",
    "build_nsgaii_permutation_config_space",
    "build_nsgaii_mixed_config_space",
    "build_nsgaii_binary_config_space",
    "build_nsgaii_integer_config_space",
    "config_from_assignment",
]
