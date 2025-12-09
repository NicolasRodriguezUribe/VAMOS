"""Deprecated shim for vamos.tuning.sampler."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.sampler' is deprecated; use 'vamos.tuning.core.sampler' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .core.sampler import *  # noqa: F401,F403

