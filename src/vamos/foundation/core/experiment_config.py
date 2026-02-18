from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib.util import find_spec

# Constants moved from runner.py
TITLE = "VAMOS Experiment Runner"
DEFAULT_ALGORITHM = "nsgaii"
DEFAULT_ENGINE = "numpy"
DEFAULT_PROBLEM = "zdt1"

ENABLED_ALGORITHMS = ("nsgaii", "moead", "smsemoa", "nsgaiii", "spea2", "ibea", "smpso", "agemoea", "rvea")
OPTIONAL_ALGORITHMS: tuple[str, ...] = ()
EXTERNAL_ALGORITHM_NAMES = ("pymoo_nsga2", "jmetalpy_nsga2", "pygmo_nsga2")
HV_REFERENCE_OFFSET = 0.1
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

    If engine is None or "auto", prefer numba for selected algorithms when available;
    otherwise fall back to DEFAULT_ENGINE.
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


@dataclass
class ExperimentConfig:
    title: str = TITLE
    # Capture the environment at instantiation time so test fixtures that tweak
    # VAMOS_OUTPUT_ROOT take effect even if the module was imported earlier.
    output_root: str = field(default_factory=lambda: os.environ.get("VAMOS_OUTPUT_ROOT", "results"))
    population_size: int = 100
    offspring_population_size: int | None = None
    max_evaluations: int = 25000
    seed: int = 42
    eval_strategy: str = "serial"
    n_workers: int | None = None
    live_viz: bool = False
    live_viz_interval: int = 5
    live_viz_max_points: int = 1000

    def offspring_size(self) -> int:
        if self.offspring_population_size is not None:
            return self.offspring_population_size
        return self.population_size
