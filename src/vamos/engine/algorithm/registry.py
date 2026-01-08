"""
Algorithm registry.

Maps algorithm names to builder callables so orchestration code avoids
hard-coded conditionals. Builders are expected to accept (config_dict, kernel)
and return an initialized algorithm instance.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol

from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol

from .nsgaii import NSGAII
from .nsgaiii import NSGAIII
from .moead import MOEAD
from .smsemoa import SMSEMOA
from .spea2 import SPEA2
from .ibea import IBEA
from .smpso import SMPSO
from vamos.foundation.registry import Registry


class AlgorithmLike(Protocol):
    def run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any],
        seed: int,
        eval_backend: Any | None = None,
        live_viz: Any | None = None,
    ) -> Mapping[str, Any]: ...


AlgorithmBuilder = Callable[[dict[str, Any], KernelBackend], AlgorithmLike]

_ALGORITHMS: Registry[AlgorithmBuilder] | None = None


def _build_nsgaii(cfg: dict[str, Any], kernel: KernelBackend) -> AlgorithmLike:
    return NSGAII(cfg, kernel=kernel)


def _build_nsgaiii(cfg: dict[str, Any], kernel: KernelBackend) -> AlgorithmLike:
    return NSGAIII(cfg, kernel=kernel)


def _build_moead(cfg: dict[str, Any], kernel: KernelBackend) -> AlgorithmLike:
    return MOEAD(cfg, kernel=kernel)


def _build_smsemoa(cfg: dict[str, Any], kernel: KernelBackend) -> AlgorithmLike:
    return SMSEMOA(cfg, kernel=kernel)


def _build_spea2(cfg: dict[str, Any], kernel: KernelBackend) -> AlgorithmLike:
    return SPEA2(cfg, kernel=kernel)


def _build_ibea(cfg: dict[str, Any], kernel: KernelBackend) -> AlgorithmLike:
    return IBEA(cfg, kernel=kernel)


def _build_smpso(cfg: dict[str, Any], kernel: KernelBackend) -> AlgorithmLike:
    """
    Note: SMPSO typically requires a different config structure, but the builder
    signature remains consistent.
    """
    return SMPSO(cfg, kernel=kernel)


def _build_agemoea(cfg: dict[str, Any], kernel: KernelBackend) -> AlgorithmLike:
    from .agemoea import AGEMOEA

    return AGEMOEA(cfg, kernel=kernel)


def _build_rvea(cfg: dict[str, Any], kernel: KernelBackend) -> AlgorithmLike:
    from .rvea import RVEA

    return RVEA(cfg, kernel=kernel)


def _register_algorithms(registry: Registry[AlgorithmBuilder]) -> None:
    registry.register("nsgaii", _build_nsgaii)
    registry.register("nsgaiii", _build_nsgaiii)
    registry.register("moead", _build_moead)
    registry.register("smsemoa", _build_smsemoa)
    registry.register("spea2", _build_spea2)
    registry.register("ibea", _build_ibea)
    registry.register("smpso", _build_smpso)
    registry.register("agemoea", _build_agemoea)
    registry.register("rvea", _build_rvea)


def get_algorithms_registry() -> Registry[AlgorithmBuilder]:
    global _ALGORITHMS
    if _ALGORITHMS is None:
        registry: Registry[AlgorithmBuilder] = Registry("Algorithms")
        _register_algorithms(registry)
        _ALGORITHMS = registry
    return _ALGORITHMS


def resolve_algorithm(name: str) -> AlgorithmBuilder:
    key = name.lower()
    registry = get_algorithms_registry()
    try:
        return registry[key]
    except KeyError as exc:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unsupported algorithm '{name}'. Available: {available}") from exc


def __getattr__(name: str) -> Any:
    if name == "ALGORITHMS":
        return get_algorithms_registry()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["get_algorithms_registry", "resolve_algorithm", "AlgorithmBuilder", "AlgorithmLike"]
