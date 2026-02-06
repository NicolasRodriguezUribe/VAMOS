from __future__ import annotations

from vamos.algorithms import available_algorithms, available_crossover_methods, available_mutation_methods
from vamos.experiment.cli.quickstart import available_templates
from vamos.foundation.kernel.registry import KERNELS


def build_catalog(problem_type: str = "real") -> dict[str, list[str]]:
    """Build a deterministic catalog of available optimization building blocks."""
    return {
        "algorithms": sorted(available_algorithms()),
        "kernels": sorted(KERNELS.keys()),
        "crossover_methods": sorted(available_crossover_methods(problem_type)),
        "mutation_methods": sorted(available_mutation_methods(problem_type)),
        "templates": sorted(available_templates()),
    }


__all__ = ["build_catalog"]
