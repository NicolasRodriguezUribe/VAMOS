"""Deprecated shim for vamos.tuning.bridge."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.bridge' is deprecated; use 'vamos.tuning.racing.bridge' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .racing.bridge import *  # noqa: F401,F403

