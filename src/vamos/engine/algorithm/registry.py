"""
Algorithm registry.

Maps algorithm names to builder callables so orchestration code avoids
hard-coded conditionals. Builders are expected to accept (config_dict, kernel)
and return an initialized algorithm instance.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Any, Protocol

from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol

from .nsgaii import NSGAII
from .nsgaiii import NSGAIII
from .moead import MOEAD
from .smsemoa import SMSEMOA
from .spea2 import SPEA2
from .ibea import IBEA
from .smpso import SMPSO


class AlgorithmLike(Protocol):
    def run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any],
        seed: int,
        eval_backend: Any | None = None,
        live_viz: Any | None = None,
    ) -> Mapping[str, Any]: ...


AlgorithmBuilder = Callable[[dict, KernelBackend], AlgorithmLike]

from vamos.foundation.registry import Registry

ALGORITHMS: Registry[AlgorithmBuilder] = Registry("Algorithms")


@ALGORITHMS.register("nsgaii")
def _build_nsgaii(cfg: dict, kernel: KernelBackend) -> AlgorithmLike:
    return NSGAII(cfg, kernel=kernel)


@ALGORITHMS.register("nsgaiii")
def _build_nsgaiii(cfg: dict, kernel: KernelBackend) -> AlgorithmLike:
    return NSGAIII(cfg, kernel=kernel)


@ALGORITHMS.register("moead")
def _build_moead(cfg: dict, kernel: KernelBackend) -> AlgorithmLike:
    return MOEAD(cfg, kernel=kernel)


@ALGORITHMS.register("smsemoa")
def _build_smsemoa(cfg: dict, kernel: KernelBackend) -> AlgorithmLike:
    return SMSEMOA(cfg, kernel=kernel)


@ALGORITHMS.register("spea2")
def _build_spea2(cfg: dict, kernel: KernelBackend) -> AlgorithmLike:
    return SPEA2(cfg, kernel=kernel)


@ALGORITHMS.register("ibea")
def _build_ibea(cfg: dict, kernel: KernelBackend) -> AlgorithmLike:
    return IBEA(cfg, kernel=kernel)


@ALGORITHMS.register("smpso")
def _build_smpso(cfg: dict, kernel: KernelBackend) -> AlgorithmLike:
    """
    Note: SMPSO typically requires a different config structure, but the builder
    signature remains consistent.
    """
    return SMPSO(cfg, kernel=kernel)


@ALGORITHMS.register("agemoea")
def _build_agemoea(cfg: dict, kernel: KernelBackend) -> AlgorithmLike:
    from .agemoea import AGEMOEA
    return AGEMOEA(cfg, kernel=kernel)


@ALGORITHMS.register("rvea")
def _build_rvea(cfg: dict, kernel: KernelBackend) -> AlgorithmLike:
    from .rvea import RVEA
    return RVEA(cfg, kernel=kernel)


def resolve_algorithm(name: str) -> AlgorithmBuilder:
    key = name.lower()
    try:
        return ALGORITHMS[key]
    except KeyError as exc:
        available = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unsupported algorithm '{name}'. Available: {available}") from exc


__all__ = ["ALGORITHMS", "resolve_algorithm", "AlgorithmBuilder", "AlgorithmLike"]
