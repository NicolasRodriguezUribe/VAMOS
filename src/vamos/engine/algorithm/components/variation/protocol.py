from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VariationOperator(Protocol):
    """
    Protocol for any variation operator (crossover, mutation, repair).
    """

    def __call__(self, X: np.ndarray, rng: np.random.Generator, **kwargs) -> np.ndarray:
        """
        Apply the operator to population X.

        Args:
            X: Input population/offspring array.
            rng: Random number generator.
            **kwargs: Additional context (e.g. current generation).

        Returns:
            Modified population array.
        """
        ...
