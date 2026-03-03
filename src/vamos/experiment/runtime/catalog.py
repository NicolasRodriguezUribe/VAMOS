from __future__ import annotations

from importlib.util import find_spec

TITLE = "VAMOS Experiment Runner"
DEFAULT_ENGINE = "numpy"
DEFAULT_PROBLEM = "zdt1"

EXTERNAL_ALGORITHM_NAMES = ("pymoo_nsga2", "jmetalpy_nsga2", "pygmo_nsga2")
EXPERIMENT_TYPES = ("backends",)
EXPERIMENT_BACKENDS = (
    "numpy",
    "numba",
    "moocore",
    "jax",
)

_PREFER_NUMBA_ALGORITHMS = {"nsgaii", "moead"}


def _has_numba() -> bool:
    return find_spec("numba") is not None


def resolve_engine(engine: str | None, *, algorithm: str | None = None) -> str:
    """
    Resolve the effective engine for a run.

    If engine is None or "auto", prefer numba for selected algorithms when
    available; otherwise fall back to the default engine.
    """
    if engine is None:
        if algorithm and algorithm.lower() in _PREFER_NUMBA_ALGORITHMS and _has_numba():
            return "numba"
        return DEFAULT_ENGINE
    engine_name = str(engine).lower()
    if engine_name == "auto":
        if algorithm and algorithm.lower() in _PREFER_NUMBA_ALGORITHMS and _has_numba():
            return "numba"
        return DEFAULT_ENGINE
    return engine_name


__all__ = [
    "TITLE",
    "DEFAULT_ENGINE",
    "DEFAULT_PROBLEM",
    "EXTERNAL_ALGORITHM_NAMES",
    "EXPERIMENT_TYPES",
    "EXPERIMENT_BACKENDS",
    "resolve_engine",
]
