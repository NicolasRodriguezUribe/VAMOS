"""Shared base classes for real-valued intensification operators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .utils import ArrayLike, RealOperator


class IntensificationOperator(RealOperator, ABC):
    """Base class for real-coded intensification operators."""

    @abstractmethod
    def __call__(
        self,
        offspring: ArrayLike,
        rng: np.random.Generator,
        *,
        parents: ArrayLike | None = None,
    ) -> ArrayLike:
        raise NotImplementedError


__all__ = ["IntensificationOperator"]
