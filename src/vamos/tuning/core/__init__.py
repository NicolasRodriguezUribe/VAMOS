"""Shared/core utilities for VAMOS tuning."""

from .param_space import ParamSpace, Real, Int, Categorical, Condition
from .parameter_space import (
    AlgorithmConfigSpace,
    Boolean,
    Categorical,
    CategoricalInteger,
    Double,
    Integer,
    ParameterDefinition,
)
from .config_space import AlgorithmConfigSpace as LegacyAlgorithmConfigSpace
from .parameters import (
    BaseParam,
    CategoricalIntegerParam,
    CategoricalParam,
    IntegerParam,
    FloatParam,
    BooleanParam,
    ConditionalBlock,
)
from .sampler import Sampler, UniformSampler, ModelBasedSampler
from .tuning_task import TuningTask, EvalContext, Instance
from .scenario import Scenario
from .history import (
    TrialRecord,
    JsonLike,
    load_history_json,
    load_histories_from_directory,
    select_top_k_trials,
    make_config_specs_from_trials,
    load_top_k_as_config_specs,
)
from .io import filter_active_config, history_to_dict, save_history_json, save_history_csv
from .validation import (
    BenchmarkSuite,
    ConfigSpec,
    RunResult,
    BenchmarkReport,
    ConfigSummary,
    ConfigInstanceSummary,
    StatisticalComparisonResult,
    run_benchmark_suite,
    summarize_benchmark,
    summarize_benchmark_per_instance,
    select_significantly_worse_configs,
)
from .stats import build_score_matrix, select_configs_by_paired_test, _z_critical

__all__ = [
    "ParamSpace",
    "Real",
    "Int",
    "Categorical",
    "Condition",
    "AlgorithmConfigSpace",
    "CategoricalInteger",
    "Double",
    "Integer",
    "Boolean",
    "ParameterDefinition",
    "LegacyAlgorithmConfigSpace",
    "BaseParam",
    "CategoricalParam",
    "CategoricalIntegerParam",
    "IntegerParam",
    "FloatParam",
    "BooleanParam",
    "ConditionalBlock",
    "Sampler",
    "UniformSampler",
    "ModelBasedSampler",
    "TuningTask",
    "EvalContext",
    "Instance",
    "Scenario",
    "TrialRecord",
    "JsonLike",
    "load_history_json",
    "load_histories_from_directory",
    "select_top_k_trials",
    "make_config_specs_from_trials",
    "load_top_k_as_config_specs",
    "filter_active_config",
    "history_to_dict",
    "save_history_json",
    "save_history_csv",
    "BenchmarkSuite",
    "ConfigSpec",
    "RunResult",
    "BenchmarkReport",
    "ConfigSummary",
    "ConfigInstanceSummary",
    "StatisticalComparisonResult",
    "run_benchmark_suite",
    "summarize_benchmark",
    "summarize_benchmark_per_instance",
    "select_significantly_worse_configs",
    "build_score_matrix",
    "select_configs_by_paired_test",
    "_z_critical",
]
