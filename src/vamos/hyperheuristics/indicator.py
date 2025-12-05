from __future__ import annotations

import numpy as np

from vamos.algorithm.hypervolume import hypervolume

try:
    from vamos.metrics.moocore_indicators import get_indicator
except Exception:  # pragma: no cover - optional moocore dependency
    get_indicator = None


class IndicatorEvaluator:
    """
    Lightweight helper to evaluate indicators for reward computation.
    Supports 'hv' and a subset of MooCore indicators when available.
    """

    def __init__(self, name: str, reference_point: np.ndarray | None = None, mode: str = "maximize"):
        self.name = name.lower()
        self.mode = mode
        self.reference_point = reference_point
        self._indicator = None
        if self.name.startswith("igd") or self.name.startswith("epsilon"):
            if get_indicator is None:
                raise ImportError("MooCore indicators are not available; install moocore to enable IGD/epsilon.")
            self._indicator = get_indicator(self.name)

    def compute(self, F: np.ndarray) -> float:
        if self.name in {"hv", "hypervolume"}:
            if self.reference_point is None:
                ref = np.max(F, axis=0) + 0.1
            else:
                ref = self.reference_point
            return float(hypervolume(F, ref))
        if self._indicator is not None:
            return float(self._indicator.compute(F).value)
        raise ValueError(f"Unsupported indicator '{self.name}'.")

