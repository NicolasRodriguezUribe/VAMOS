"""Racing tuner package with unified parameter types."""

from .bridge import (
    build_agemoea_config_space,
    build_ibea_binary_config_space,
    build_ibea_config_space,
    build_ibea_integer_config_space,
    build_moead_binary_config_space,
    build_moead_config_space,
    build_moead_integer_config_space,
    build_moead_permutation_config_space,
    build_nsgaii_binary_config_space,
    build_nsgaii_config_space,
    build_nsgaii_integer_config_space,
    build_nsgaii_mixed_config_space,
    build_nsgaii_permutation_config_space,
    build_nsgaiii_binary_config_space,
    build_nsgaiii_config_space,
    build_nsgaiii_integer_config_space,
    build_rvea_config_space,
    build_smpso_config_space,
    build_smsemoa_binary_config_space,
    build_smsemoa_config_space,
    build_smsemoa_integer_config_space,
    build_spea2_config_space,
    config_from_assignment,
)
from .config_space import AlgorithmConfigSpace
from .core import RacingTuner
from .io import filter_active_config, history_to_dict, load_checkpoint, save_checkpoint, save_history_csv, save_history_json
from .param_space import (
    Boolean,
    BooleanParam,
    Categorical,
    CategoricalIntegerParam,
    CategoricalParam,
    Condition,
    ConditionalBlock,
    # Aliases
    FloatParam,
    Int,
    IntegerParam,
    ParamSpace,
    ParamType,
    Real,
)
from .parameters import BaseParam
from .random_search_tuner import RandomSearchTuner, TrialResult
from .sampler import ModelBasedSampler, Sampler, UniformSampler
from .scenario import Scenario
from .state import ConfigState, EliteEntry
from .tuning_task import EvalContext, Instance, TuningTask
from .two_phase import TwoPhaseScenario, TwoPhaseTuner
from .warm_start import WarmStartEvaluator

__all__ = [
    # State
    "ConfigState",
    "EliteEntry",
    # Tuners
    "RacingTuner",
    "RandomSearchTuner",
    "TrialResult",
    # Parameter types
    "ParamSpace",
    "Real",
    "Int",
    "Categorical",
    "Boolean",
    "Condition",
    "ConditionalBlock",
    "ParamType",
    # Aliases
    "FloatParam",
    "IntegerParam",
    "CategoricalParam",
    "BooleanParam",
    "CategoricalIntegerParam",
    "BaseParam",
    # Samplers
    "Sampler",
    "UniformSampler",
    "ModelBasedSampler",
    # Task/Scenario
    "TuningTask",
    "EvalContext",
    "Instance",
    "Scenario",
    # Config space
    "AlgorithmConfigSpace",
    # Warm-start helper
    "WarmStartEvaluator",
    # Two-phase tuner
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
