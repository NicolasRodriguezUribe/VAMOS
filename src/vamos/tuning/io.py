"""Deprecated shim for vamos.tuning.io."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.io' is deprecated; use 'vamos.tuning.core.io' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .core.io import *  # noqa: F401,F403

