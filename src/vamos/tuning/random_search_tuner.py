"""Deprecated shim for vamos.tuning.random_search_tuner."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.random_search_tuner' is deprecated; use 'vamos.tuning.racing.random_search_tuner' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .racing.random_search_tuner import *  # noqa: F401,F403

