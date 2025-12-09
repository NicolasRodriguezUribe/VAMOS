"""Compatibility wrapper for NSGA-II config space builder."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.nsgaii' is deprecated; use 'vamos.tuning.evolver.nsgaii' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .evolver.nsgaii import build_nsgaii_config_space

__all__ = ["build_nsgaii_config_space"]
