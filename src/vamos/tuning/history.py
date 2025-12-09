"""Deprecated shim for vamos.tuning.history."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.history' is deprecated; use 'vamos.tuning.core.history' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .core.history import *  # noqa: F401,F403

