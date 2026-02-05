from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

from .tuning_task import EvalContext

EvalResult: TypeAlias = float | tuple[float, Any]
EvalFn: TypeAlias = Callable[[dict[str, Any], EvalContext], EvalResult]

__all__ = ["EvalFn", "EvalResult"]
