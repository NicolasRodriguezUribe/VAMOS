from __future__ import annotations

import numpy as np

from vamos.foundation.metrics.hypervolume import hypervolume


class HVTracker:
    def __init__(self, hv_config: dict | None, kernel):
        self.enabled = hv_config is not None
        self.last_value: float | None = None
        if not self.enabled:
            return
        self.target = float(hv_config["target_value"])
        self.ref_point = np.asarray(hv_config["reference_point"], dtype=float)
        if kernel is not None and kernel.supports_quality_indicator("hypervolume"):
            self.evaluator = kernel.hypervolume
        else:
            self.evaluator = hypervolume

    def reached(self, points: np.ndarray) -> bool:
        if not self.enabled:
            return False
        hv_val = self.evaluator(points, self.ref_point)
        self.last_value = hv_val
        return hv_val >= self.target


__all__ = ["HVTracker"]
