"""Deprecated shim for vamos.tuning.parameters."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.parameters' is deprecated; use 'vamos.tuning.core.parameters' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .core.parameters import *  # noqa: F401,F403

