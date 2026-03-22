from __future__ import annotations

from typing import Any

import numpy as np

from ._mixed_crossover import mixed_crossover
from ._mixed_init import mixed_initialize
from ._mixed_mutation import mixed_mutation


class MixedCrossover:
    def __init__(self, prob: float = 0.9, **kwargs: Any) -> None:
        self.prob = float(prob)

    def __call__(self, parents: np.ndarray, rng: np.random.Generator, **kwargs: Any) -> np.ndarray:
        spec = kwargs.get("spec")
        if spec is None:
            raise ValueError("MixedCrossover requires 'spec' in kwargs.")
        return mixed_crossover(parents, self.prob, spec, rng)


class MixedMutation:
    def __init__(self, prob: float = 0.1, **kwargs: Any) -> None:
        self.prob = float(prob)

    def __call__(self, X: np.ndarray, rng: np.random.Generator, **kwargs: Any) -> None:
        spec = kwargs.get("spec")
        if spec is None:
            raise ValueError("MixedMutation requires 'spec' in kwargs.")
        mixed_mutation(X, self.prob, spec, rng)


__all__ = [
    "mixed_initialize",
    "mixed_crossover",
    "mixed_mutation",
    "MixedCrossover",
    "MixedMutation",
]
