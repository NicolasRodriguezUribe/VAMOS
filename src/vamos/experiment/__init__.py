"""
Experiment layer: CLI entry points, study orchestration, benchmark suites, diagnostics, and curated experiment zoo.
"""

from .context import Experiment, experiment, RunRecord, ExperimentSummary

__all__ = [
    "Experiment",
    "experiment",
    "RunRecord",
    "ExperimentSummary",
]
