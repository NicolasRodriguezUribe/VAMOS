from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import numpy as np

from vamos.foundation.quality_indicators.hypervolume import hypervolume

_get_indicator: Callable[..., Any] | None
try:
    from vamos.foundation.quality_indicators.moocore_indicators import get_indicator as _get_indicator
except Exception:  # pragma: no cover - optional moocore dependency
    _get_indicator = None

get_indicator: Callable[..., Any] | None = _get_indicator

IndicatorMode: TypeAlias = Literal["maximize", "minimize"]


class IndicatorEvaluator:
    """
    Lightweight helper to evaluate indicators for reward computation.
    Supports 'hv' and a subset of MooCore indicators when available.
    """

    def __init__(
        self,
        name: str,
        reference_point: np.ndarray | None = None,
        reference_front: np.ndarray | None = None,
        mode: IndicatorMode = "maximize",
    ):
        self.name = name.lower()
        if mode not in {"maximize", "minimize"}:
            raise ValueError("mode must be one of: maximize, minimize")
        self.mode = mode
        self.reference_point = None if reference_point is None else np.asarray(reference_point, dtype=float)
        self.reference_front = None if reference_front is None else np.asarray(reference_front, dtype=float)
        self._indicator = None
        if self.name.startswith("igd") or self.name.startswith("epsilon"):
            if self.reference_front is None:
                raise ValueError(f"Indicator '{self.name}' requires reference_front.")
            if get_indicator is None:
                raise ImportError("MooCore indicators are not available; install moocore to enable IGD/epsilon.")
            self._indicator = get_indicator(self.name, reference_front=self.reference_front)

    def _apply_mode(self, value: float) -> float:
        return float(value if self.mode == "maximize" else -value)

    def compute(self, F: np.ndarray) -> float:
        if self.name in {"hv", "hypervolume"}:
            if self.reference_point is None:
                raise ValueError("Hypervolume indicator requires reference_point.")
            return self._apply_mode(float(hypervolume(F, self.reference_point)))
        if self._indicator is not None:
            return self._apply_mode(float(self._indicator.compute(F).value))
        raise ValueError(f"Unsupported indicator '{self.name}'.")


__all__ = ["IndicatorEvaluator", "IndicatorMode", "get_indicator"]
