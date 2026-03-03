from __future__ import annotations

import os
from dataclasses import dataclass, field

HV_REFERENCE_OFFSET = 0.1


@dataclass
class ExperimentConfig:
    title: str = "VAMOS Experiment Runner"
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
