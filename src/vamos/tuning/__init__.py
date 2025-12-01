"""Auto-tuning utilities for VAMOS."""

from .parameter_space import (
    AlgorithmConfigSpace,
    Boolean,
    Categorical,
    CategoricalInteger,
    Double,
    Integer,
    ParameterDefinition,
)
from .meta_problem import MetaOptimizationProblem
from .nsga2_meta import MetaNSGAII
from .pipeline import TuningPipeline, compute_hyperparameter_importance
from .tuner import NSGAIITuner
from .param_space import ParamSpace, Real, Int, Categorical, Condition
from .tuning_task import TuningTask, EvalContext, Instance
from .random_search_tuner import RandomSearchTuner, TrialResult
from .scenario import Scenario
from .racing import RacingTuner, ConfigState, EliteEntry
from .sampler import Sampler, UniformSampler, ModelBasedSampler
from .io import filter_active_config, history_to_dict, save_history_json, save_history_csv
from .nsgaii import build_nsgaii_config_space
from .validation import (
    BenchmarkSuite,
    ConfigSpec,
    RunResult as BenchmarkRunResult,
    BenchmarkReport,
    ConfigSummary,
    ConfigInstanceSummary,
    StatisticalComparisonResult,
    run_benchmark_suite,
    summarize_benchmark,
    summarize_benchmark_per_instance,
    select_significantly_worse_configs,
)
from .experiment import (
    TunerKind,
    ExperimentResult,
    TuningExperiment,
    create_random_experiment,
    create_racing_experiment,
)
from .history import (
    TrialRecord as HistoryTrialRecord,
    load_history_json,
    load_histories_from_directory,
    select_top_k_trials,
    make_config_specs_from_trials,
    load_top_k_as_config_specs,
)
from .spec import (
    SamplerSpec,
    RandomTunerSpec,
    RacingTunerSpec,
    BaselineSpec,
    HistoryBaselineSpec,
    ValidationSpec,
    ExperimentSpec,
    load_experiment_spec,
    build_experiment_from_spec,
    run_experiment_from_file,
)
import numpy as _np
from pathlib import Path as _Path
import json as _json


def _config_to_serializable(cfg):
    """Convert config objects to JSON-friendly structures."""
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return cfg


def tune(
    problem_selection,
    base_config,
    config_space,
    *,
    n_generations: int = 10,
    population_size: int = 20,
    seed: int | None = None,
    output_dir: str | None = None,
    verbose: bool = False,
):
    """
    Convenience wrapper around NSGAIITuner for a single problem.
    Returns a dict with best config, trials, and diagnostics.
    """
    problem = problem_selection.instantiate()
    tuner = NSGAIITuner(
        config_space=config_space,
        problems=[problem],
        ref_fronts=[None],
        indicators=["hv"],
        max_evals_per_problem=base_config.max_evaluations,
        n_runs_per_problem=1,
        engine="numpy",
        meta_population_size=population_size,
        meta_max_evals=max(1, n_generations) * population_size,
        seed=seed,
    )
    X_meta, F_meta, configs, diagnostics = tuner.optimize()
    configs_serializable = [_config_to_serializable(cfg) for cfg in configs]
    # Meta objectives are minimized; quality is negative HV in slot 0.
    best_idx = int(_np.argmin(F_meta[:, 0]))
    best = {
        "objective": float(F_meta[best_idx, 0]),
        "config": configs_serializable[best_idx],
        "vector": X_meta[best_idx].tolist(),
    }
    trials = []
    for vec, obj, cfg in zip(X_meta, F_meta, configs_serializable):
        trials.append(
            {
                "vector": vec.tolist(),
                "objective": obj.tolist(),
                "config": cfg,
            }
        )
    result = {"best": best, "trials": trials, "diagnostics": diagnostics}
    if output_dir:
        out_path = _Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with (out_path / "tuning_results.json").open("w", encoding="utf-8") as fh:
            _json.dump(result, fh, indent=2)
        if verbose:
            print(f"Saved tuning_results.json to {out_path}")
    return result

__all__ = [
    "AlgorithmConfigSpace",
    "Boolean",
    "Categorical",
    "CategoricalInteger",
    "Double",
    "Integer",
    "ParameterDefinition",
    "MetaOptimizationProblem",
    "MetaNSGAII",
    "TuningPipeline",
    "compute_hyperparameter_importance",
    "NSGAIITuner",
    "tune",
    "build_nsgaii_config_space",
    "ParamSpace",
    "Real",
    "Int",
    "Categorical",
    "Condition",
    "TuningTask",
    "EvalContext",
    "Instance",
    "RandomSearchTuner",
    "RacingTuner",
    "Scenario",
    "ConfigState",
    "EliteEntry",
    "TrialResult",
    "Sampler",
    "UniformSampler",
    "ModelBasedSampler",
    "BenchmarkSuite",
    "ConfigSpec",
    "BenchmarkRunResult",
    "BenchmarkReport",
    "ConfigSummary",
    "ConfigInstanceSummary",
    "StatisticalComparisonResult",
    "run_benchmark_suite",
    "summarize_benchmark",
    "summarize_benchmark_per_instance",
    "select_significantly_worse_configs",
    "TunerKind",
    "ExperimentResult",
    "TuningExperiment",
    "create_random_experiment",
    "create_racing_experiment",
    "HistoryTrialRecord",
    "load_history_json",
    "load_histories_from_directory",
    "select_top_k_trials",
    "make_config_specs_from_trials",
    "load_top_k_as_config_specs",
    "SamplerSpec",
    "RandomTunerSpec",
    "RacingTunerSpec",
    "BaselineSpec",
    "HistoryBaselineSpec",
    "ValidationSpec",
    "ExperimentSpec",
    "load_experiment_spec",
    "build_experiment_from_spec",
    "run_experiment_from_file",
    "filter_active_config",
    "history_to_dict",
    "save_history_json",
    "save_history_csv",
]
