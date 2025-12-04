from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Tuple

from vamos.algorithm.config import NSGAIIConfigData
from vamos.algorithm.nsgaii import NSGAII
from vamos.kernel.numpy_backend import NumPyKernel
from vamos.kernel.registry import resolve_kernel
from vamos.problem.types import ProblemProtocol


class OptimizationResult:
    """Simple container returned by optimize()."""

    def __init__(self, payload: Mapping[str, Any]):
        self.F = payload.get("F")
        self.X = payload.get("X")
        self.data = dict(payload)


def optimize(
    problem: ProblemProtocol,
    config: Any,
    termination: Tuple[str, Any],
    seed: int,
    engine: str = "numpy",
) -> OptimizationResult:
    """
    Run a single optimization for the provided problem/config pair.

    Args:
        problem: A problem implementing the ProblemProtocol (evaluate, xl/xu, n_var/n_obj).
        config: Algorithm config dataclass/builder or mapping compatible with NSGA-II for now.
        termination: Tuple describing termination criterion, e.g. ("n_eval", 1000).
        seed: RNG seed used by the algorithm.
        engine: Kernel backend name ("numpy", "numba", "moocore").
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
