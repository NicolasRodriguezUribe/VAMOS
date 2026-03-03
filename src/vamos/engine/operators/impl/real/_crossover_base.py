"""Shared base classes for real-valued crossover operators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .utils import ArrayLike, RealOperator


class Crossover(RealOperator, ABC):
    """Base class for real-coded crossover operators."""

    @abstractmethod
    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        raise NotImplementedError


__all__ = ["Crossover"]
