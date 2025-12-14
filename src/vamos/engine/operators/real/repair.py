"""Bounds repair helpers for real-encoded individuals."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .utils import ArrayLike, RealOperator, _clip_population, _ensure_bounds


class Repair(RealOperator, ABC):
    """Base class for repair operators (bounds and feasibility helpers)."""

    @abstractmethod
    def __call__(
        self,
        x: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator,
    ) -> ArrayLike:
        raise NotImplementedError


class ClampRepair(Repair):
    """Repair operator that clamps variables to their bounds."""

    def __call__(
        self,
        x: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator,  # pylint: disable=unused-argument
    ) -> ArrayLike:
        lower_arr, upper_arr = _ensure_bounds(lower, upper)
        values = self._as_population(x, name="x")
        return _clip_population(values, lower_arr, upper_arr)


class ReflectRepair(Repair):
    """Repair operator that reflects out-of-bounds values back into range."""

    def __call__(
        self,
        x: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator,  # pylint: disable=unused-argument
    ) -> ArrayLike:
        lower_arr, upper_arr = _ensure_bounds(lower, upper)
        result = self._as_population(x, name="x")
        n_vars = lower_arr.shape[0]
        for j in range(n_vars):
            low = lower_arr[j]
            high = upper_arr[j]
            width = high - low
            if width <= 0.0:
                result[:, j] = low
                continue
            val = result[:, j] - low
            period = 2.0 * width
            val = np.mod(val, period)
            over = val > width
            val[over] = period - val[over]
            result[:, j] = val + low
        return result


class ResampleRepair(Repair):
    """Repair operator that resamples violated genes uniformly inside bounds."""

    def __call__(
        self,
        x: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator,
    ) -> ArrayLike:
        lower_arr, upper_arr = _ensure_bounds(lower, upper)
        result = self._as_population(x, name="x")
        mask_low = result < lower_arr
        mask_high = result > upper_arr
        mask = mask_low | mask_high
        if not np.any(mask):
            return result
        span = upper_arr - lower_arr
        resampled = lower_arr + rng.random(result.shape) * span
        result[mask] = resampled[mask]
        return result


class RoundRepair(Repair):
    """Repair operator that rounds genes to the nearest integer inside the bounds."""

    def __call__(
        self,
        x: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        rng: np.random.Generator,  # pylint: disable=unused-argument
    ) -> ArrayLike:
        lower_arr, upper_arr = _ensure_bounds(lower, upper)
        rounded = np.rint(self._as_population(x, name="x"))
        return _clip_population(rounded, lower_arr, upper_arr)


__all__ = [
    "ClampRepair",
    "ReflectRepair",
    "Repair",
    "ResampleRepair",
    "RoundRepair",
]
