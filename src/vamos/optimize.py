from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Tuple

from vamos.algorithm.config import NSGAIIConfigData, SPEA2ConfigData, IBEAConfigData, SMPSOConfigData
from vamos.algorithm.nsgaii import NSGAII
from vamos.algorithm.spea2 import SPEA2
from vamos.algorithm.ibea import IBEA
from vamos.algorithm.smpso import SMPSO
from vamos.kernel.numpy_backend import NumPyKernel
from vamos.kernel.registry import resolve_kernel
from vamos.problem.types import ProblemProtocol
from vamos.eval.backends import resolve_eval_backend


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
    eval_backend=None,
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

    if eval_backend is not None:
        backend = eval_backend
    else:
        backend_name = cfg_dict["eval_backend"] if isinstance(cfg_dict, dict) and "eval_backend" in cfg_dict else "serial"
        backend = resolve_eval_backend(backend_name)

    if isinstance(config, NSGAIIConfigData) or cfg_dict.get("survival") == "nsga2":
        algorithm = NSGAII(cfg_dict, kernel=kernel)
    elif isinstance(config, SPEA2ConfigData) or cfg_dict.get("k_neighbors") is not None:
        algorithm = SPEA2(cfg_dict, kernel=kernel)
    elif isinstance(config, IBEAConfigData) or cfg_dict.get("indicator") is not None:
        algorithm = IBEA(cfg_dict, kernel=kernel)
    elif isinstance(config, SMPSOConfigData) or any(key in cfg_dict for key in ("inertia", "c1", "c2")):
        algorithm = SMPSO(cfg_dict, kernel=kernel)
    else:
        raise ValueError("optimize() currently supports NSGA-II, SPEA2, IBEA, and SMPSO configurations.")

    run_fn = getattr(algorithm, "run")
    import inspect

    if "eval_backend" in inspect.signature(run_fn).parameters:
        result = run_fn(problem, termination=termination, seed=seed, eval_backend=backend)
    else:
        result = run_fn(problem, termination=termination, seed=seed)
    return OptimizationResult(result)
