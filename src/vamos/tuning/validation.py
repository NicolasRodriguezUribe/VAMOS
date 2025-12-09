"""Deprecated shim for vamos.tuning.validation."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.validation' is deprecated; use 'vamos.tuning.core.validation' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .core.validation import *  # noqa: F401,F403

