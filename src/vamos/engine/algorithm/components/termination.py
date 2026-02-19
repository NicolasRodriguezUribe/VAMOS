from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from vamos.foundation.metrics.hypervolume import hypervolume


class HVTracker:
    def __init__(self, hv_config: dict[str, Any] | None, kernel: Any) -> None:
        self.enabled = hv_config is not None
        self.last_value: float | None = None
        if not self.enabled:
            return
        assert hv_config is not None
        self.target = float(hv_config["target_value"])
        self.ref_point = np.asarray(hv_config["reference_point"], dtype=float)
        if kernel is not None and kernel.supports_quality_indicator("hypervolume"):
            self.evaluator: Callable[[np.ndarray, np.ndarray], float] = kernel.hypervolume
        else:
            self.evaluator = hypervolume

    def reached(self, points: np.ndarray) -> bool:
        if not self.enabled:
            return False
        hv_val = float(self.evaluator(points, self.ref_point))
        self.last_value = hv_val
        return hv_val >= self.target


def parse_termination(
    termination: tuple[str, Any],
    algorithm_name: str = "algorithm",
) -> tuple[int, dict[str, Any] | None]:
    """
    Parse termination criterion and return (max_eval, hv_config).

    Parameters
    ----------
    termination : tuple[str, Any]
        Termination criterion as (type, value). Supported types:
        - "max_evaluations": value is the max number of evaluations
        - "hv": value is a dict with hypervolume config
    algorithm_name : str
        Algorithm name for error messages.

    Returns
    -------
    tuple[int, dict[str, Any] | None]
        (max_evaluations, hv_config or None)

    Raises
    ------
    ValueError
        If termination type is unsupported or HV config is invalid.
    """
    term_type, term_val = termination
    hv_config = None

    if term_type == "max_evaluations":
        max_eval = int(term_val)
    elif term_type == "hv":
        hv_config = dict(term_val)
        max_eval = int(hv_config.get("max_evaluations", 0))
        if max_eval <= 0:
            raise ValueError(f"HV-based termination for {algorithm_name} requires a positive max_evaluations value.")
    else:
        raise ValueError(f"Unsupported termination criterion '{term_type}' for {algorithm_name}.")

    return max_eval, hv_config


__all__ = ["HVTracker", "parse_termination"]
