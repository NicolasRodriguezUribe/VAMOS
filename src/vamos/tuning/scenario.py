"""Deprecated shim for vamos.tuning.scenario."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.scenario' is deprecated; use 'vamos.tuning.core.scenario' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .core.scenario import *  # noqa: F401,F403

