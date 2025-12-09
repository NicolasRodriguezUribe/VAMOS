"""Deprecated pit package for ParamSpace-based tuning."""

from warnings import warn

warn(
    "Importing 'vamos.tuning.pit' is deprecated; use 'vamos.tuning.core.param_space' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from vamos.tuning.core.param_space import ParamSpace, Real, Int, Categorical, Condition

__all__ = ["ParamSpace", "Real", "Int", "Categorical", "Condition"]
