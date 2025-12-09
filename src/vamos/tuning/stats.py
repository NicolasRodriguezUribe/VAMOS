"""Deprecated shim for vamos.tuning.stats."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.stats' is deprecated; use 'vamos.tuning.core.stats' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .core.stats import *  # noqa: F401,F403

