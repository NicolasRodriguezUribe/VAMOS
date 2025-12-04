from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Constants moved from runner.py
TITLE = "VAMOS Experiment Runner"
DEFAULT_ALGORITHM = "nsgaii"
DEFAULT_ENGINE = "numpy"
DEFAULT_PROBLEM = "zdt1"

ENABLED_ALGORITHMS = ("nsgaii", "moead", "smsemoa", "nsga3", "spea2", "ibea", "smpso")
OPTIONAL_ALGORITHMS: tuple[str, ...] = ()
EXTERNAL_ALGORITHM_NAMES = ("pymoo_nsga2", "jmetalpy_nsga2", "pygmo_nsga2")
HV_REFERENCE_OFFSET = 0.1

EXPERIMENT_BACKENDS = (
    "numpy",
    "numba",
    "moocore",
)

@dataclass
class ExperimentConfig:
    title: str = TITLE
    output_root: str = os.environ.get("VAMOS_OUTPUT_ROOT", "results")
    population_size: int = 100
    offspring_population_size: int | None = None
    max_evaluations: int = 25000
    seed: int = 42

    def offspring_size(self) -> int:
        if self.offspring_population_size is not None:
            return self.offspring_population_size
        return self.population_size
