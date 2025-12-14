from __future__ import annotations

"""
Lightweight facade for tuning utilities (AutoNSGA-II and config spaces).

For full control, import from `vamos.engine.tuning.*`.
"""

from vamos.engine.tuning import (
    AlgorithmConfigSpace,
    BenchmarkReport,
    BenchmarkRunResult,
    BenchmarkSuite,
    ConfigInstanceSummary,
    ConfigSpec,
    ConfigSummary,
    MetaNSGAII,
    NSGAIITuner,
    RandomSearchTuner,
    TuningPipeline,
    build_ibea_config_space,
    build_moead_config_space,
    build_nsga3_config_space,
    build_nsgaii_config_space,
    build_smpso_config_space,
    build_smsemoa_config_space,
    build_spea2_config_space,
    compute_hyperparameter_importance,
    run_benchmark_suite,
    summarize_benchmark,
)

__all__ = [
    "AlgorithmConfigSpace",
    "MetaNSGAII",
    "NSGAIITuner",
    "RandomSearchTuner",
    "TuningPipeline",
    "build_nsgaii_config_space",
    "build_nsga3_config_space",
    "build_moead_config_space",
    "build_smsemoa_config_space",
    "build_spea2_config_space",
    "build_ibea_config_space",
    "build_smpso_config_space",
    "compute_hyperparameter_importance",
    "BenchmarkSuite",
    "BenchmarkReport",
    "BenchmarkRunResult",
    "ConfigSpec",
    "ConfigSummary",
    "ConfigInstanceSummary",
    "run_benchmark_suite",
    "summarize_benchmark",
]

