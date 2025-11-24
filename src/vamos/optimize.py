from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from vamos.algorithm.config import NSGAIIConfigData
from vamos.algorithm.nsgaii import NSGAII
from vamos.kernel.numpy_backend import NumPyKernel
from vamos.runner import resolve_kernel


class OptimizationResult:
    def __init__(self, payload: dict):
        self.F = payload.get("F")
        self.X = payload.get("X")
        self.data = payload


def optimize(problem, config: Any, termination, seed: int, engine: str = "numpy") -> OptimizationResult:
    """
    Run a single optimization for the provided problem/config pair.
    """
    if hasattr(config, "to_dict"):
        cfg_dict = config.to_dict()
    elif is_dataclass(config):
        cfg_dict = asdict(config)
    else:
        cfg_dict = dict(config)
    effective_engine = engine or cfg_dict.get("engine", "numpy")
    kernel = resolve_kernel(effective_engine) if effective_engine != "numpy" else NumPyKernel()

    if isinstance(config, NSGAIIConfigData) or cfg_dict.get("survival") == "nsga2":
        algorithm = NSGAII(cfg_dict, kernel=kernel)
    else:
        raise ValueError("optimize() currently supports NSGA-II configurations only.")

    result = algorithm.run(problem, termination=termination, seed=seed)
    return OptimizationResult(result)
