"""Deprecated shim for vamos.tuning.experiment."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.experiment' is deprecated; use 'vamos.tuning.core.experiment' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .core.experiment import *  # noqa: F401,F403

