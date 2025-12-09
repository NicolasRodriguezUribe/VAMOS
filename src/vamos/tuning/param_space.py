"""Deprecated shim for the legacy param_space module."""
from warnings import warn

warn(
    "Importing 'vamos.tuning.param_space' is deprecated; use 'vamos.tuning.core.param_space' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .core.param_space import *  # noqa: F401,F403
