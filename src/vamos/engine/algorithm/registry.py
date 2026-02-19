"""
Algorithm registry.

Maps algorithm names to builder callables so orchestration code avoids
hard-coded conditionals. Builders are expected to accept (config_dict, kernel)
and return an initialized algorithm instance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from vamos.engine.algorithm.config.types import AlgorithmConfigMapping
from vamos.foundation.exceptions import _suggest_names
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol
from vamos.foundation.registry import Registry

from .ibea import IBEA
from .moead import MOEAD
from .nsgaii import NSGAII
from .nsgaiii import NSGAIII
from .smpso import SMPSO
from .smsemoa import SMSEMOA
from .spea2 import SPEA2


class AlgorithmLike(Protocol):
    def run(
        self,
        problem: ProblemProtocol,
        termination: tuple[str, Any] = ("max_evaluations", 25000),
        seed: int = 0,
        eval_strategy: Any | None = None,
        live_viz: Any | None = None,
    ) -> Mapping[str, Any]: ...


AlgorithmBuilder = Callable[[AlgorithmConfigMapping, KernelBackend | None], AlgorithmLike]

_ALGORITHMS: Registry[AlgorithmBuilder] | None = None
_ALGO_DOCS = "docs/reference/algorithms.md"
_TROUBLESHOOTING_DOCS = "docs/guide/troubleshooting.md"


def _format_unknown_algorithm(name: str, options: list[str]) -> str:
    parts = [f"Unsupported algorithm '{name}'.", f"Available: {', '.join(options)}."]
    suggestions = _suggest_names(name, options)
    if suggestions:
        if len(suggestions) == 1:
            parts.append(f"Did you mean '{suggestions[0]}'?")
        else:
            parts.append("Did you mean one of: " + ", ".join(f"'{item}'" for item in suggestions) + "?")
    parts.append(f"Docs: {_ALGO_DOCS}.")
    parts.append(f"Troubleshooting: {_TROUBLESHOOTING_DOCS}.")
    return " ".join(parts)


def _build_nsgaii(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    return NSGAII(dict(cfg), kernel=kernel)


def _build_nsgaiii(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    return NSGAIII(dict(cfg), kernel=kernel)


def _build_moead(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    return MOEAD(dict(cfg), kernel=kernel)


def _build_smsemoa(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    return SMSEMOA(dict(cfg), kernel=kernel)


def _build_spea2(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    return SPEA2(dict(cfg), kernel=kernel)


def _build_ibea(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    return IBEA(dict(cfg), kernel=kernel)


def _build_smpso(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    """
    Note: SMPSO typically requires a different config structure, but the builder
    signature remains consistent.
    """
    return SMPSO(dict(cfg), kernel=kernel)


def _build_agemoea(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    from .agemoea import AGEMOEA

    return AGEMOEA(dict(cfg), kernel=kernel)


def _build_rvea(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> AlgorithmLike:
    from .rvea import RVEA

    return RVEA(dict(cfg), kernel=kernel)


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
        available = sorted(registry.keys())
        raise ValueError(_format_unknown_algorithm(name, available)) from exc


def __getattr__(name: str) -> Any:
    if name == "ALGORITHMS":
        return get_algorithms_registry()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["get_algorithms_registry", "resolve_algorithm", "AlgorithmBuilder", "AlgorithmLike"]
