"""Compatibility wrapper for NSGA-II meta-optimizer."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.nsga2_meta' is deprecated; use 'vamos.tuning.evolver.nsga2_meta' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .evolver.nsga2_meta import MetaNSGAII

__all__ = ["MetaNSGAII"]
