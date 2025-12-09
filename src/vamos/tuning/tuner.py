"""Compatibility wrapper for NSGA-II tuner."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.tuner' is deprecated; use 'vamos.tuning.evolver.tuner' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .evolver.tuner import NSGAIITuner

__all__ = ["NSGAIITuner"]
