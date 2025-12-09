"""Deprecated shim for vamos.tuning.spec."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.spec' is deprecated; use 'vamos.tuning.core.spec' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .core.spec import *  # noqa: F401,F403

