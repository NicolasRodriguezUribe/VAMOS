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


def resolve_engine_details(engine: str | None, *, algorithm: str | None = None) -> tuple[str, str]:
    """
    Resolve the effective engine and explain where the decision came from.

    ``engine=None`` is deterministic and always uses the documented default.
    Heuristic selection is reserved for explicit ``engine="auto"``.
    """
    if engine is None:
        return DEFAULT_ENGINE, "default"
    engine_name = str(engine).lower()
    if engine_name == "auto":
        if algorithm and algorithm.lower() in _PREFER_NUMBA_ALGORITHMS and _has_numba():
            return "numba", "auto"
        return DEFAULT_ENGINE, "auto"
    return engine_name, "explicit"


def resolve_engine(engine: str | None, *, algorithm: str | None = None) -> str:
    """
    Resolve the effective engine for a run.

    ``engine=None`` is deterministic and uses the default engine.
    ``engine="auto"`` enables heuristic backend selection.
    """
    resolved_engine, _source = resolve_engine_details(engine, algorithm=algorithm)
    return resolved_engine


__all__ = [
    "TITLE",
    "DEFAULT_ENGINE",
    "DEFAULT_PROBLEM",
    "EXTERNAL_ALGORITHM_NAMES",
    "EXPERIMENT_TYPES",
    "EXPERIMENT_BACKENDS",
    "resolve_engine_details",
    "resolve_engine",
]
