"""Deprecated shim for vamos.tuning.parameter_space."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.parameter_space' is deprecated; use 'vamos.tuning.core.parameter_space' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .core.parameter_space import *  # noqa: F401,F403

