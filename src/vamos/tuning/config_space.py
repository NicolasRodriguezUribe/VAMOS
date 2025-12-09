"""Deprecated shim for vamos.tuning.config_space."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.config_space' is deprecated; use 'vamos.tuning.core.config_space' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .core.config_space import *  # noqa: F401,F403

