"""Deprecated shim for vamos.tuning.tuning_task."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.tuning_task' is deprecated; use 'vamos.tuning.core.tuning_task' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .core.tuning_task import *  # noqa: F401,F403

