"""Deprecated shim for vamos.tuning.pit.param_space."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.pit.param_space' is deprecated; use 'vamos.tuning.core.param_space' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from vamos.tuning.core.param_space import *  # noqa: F401,F403

