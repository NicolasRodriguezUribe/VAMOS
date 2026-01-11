from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class KernelBackend(ABC):
    """
    Interface for all kernel backends used by the evolutionary algorithms.
    Backends implement heavy numeric primitives on vectorized populations.
    Performance-sensitive: implementations should be vectorized/SoA and avoid Python loops.
    """

    # -------- Computation device / capability metadata --------

    def device(self) -> str:
        """
        Return a short label describing the primary execution device.
        Examples: "cpu", "gpu", "tpu".
        """
        return "cpu"

    def capabilities(self) -> Iterable[str]:
        """
        Optional backend capability tags (e.g., {"gpu", "indicator:hypervolume"}).
        """
        return ()

    def quality_indicators(self) -> Iterable[str]:
        """
        Quality indicators this backend can compute natively (e.g., {"hypervolume"}).
        """
        return ()

    def supports_quality_indicator(self, name: str) -> bool:
        normalized = name.lower()
        return any(ind.lower() == normalized for ind in self.quality_indicators())

    # -------- Evolutionary kernels --------

    @abstractmethod
    def nsga2_ranking(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return ranks and crowding distances for objective matrix F.
        """

    @abstractmethod
    def tournament_selection(
        self,
        ranks: np.ndarray,
        crowding: np.ndarray,
        pressure: int,
        rng: np.random.Generator,
        n_parents: int,
    ) -> np.ndarray:
        """
        Select n_parents indices using tournament selection.
        """

    @abstractmethod
    def sbx_crossover(
        self,
        X_parents: np.ndarray,
        params: dict[str, float],
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> np.ndarray:
        """
        Apply SBX crossover to parent array and return offspring.
        """

    @abstractmethod
    def polynomial_mutation(
        self,
        X: np.ndarray,
        params: dict[str, float],
        rng: np.random.Generator,
        xl: float,
        xu: float,
    ) -> None:
        """
        Apply polynomial mutation in-place to array X.
        """

    @abstractmethod
    def nsga2_survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        X_off: np.ndarray,
        F_off: np.ndarray,
        pop_size: int,
        return_indices: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform NSGA-II elitist replacement returning new (X, F).
        When return_indices is True, also return selected indices in the combined population.
        """

    # -------- Quality indicator hooks --------

    def hypervolume(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        """
        Optional hypervolume implementation. Backends that override this method
        should also list "hypervolume" in quality_indicators().
        """
        raise NotImplementedError(f"Backend '{self.__class__.__name__}' does not support hypervolume.")
