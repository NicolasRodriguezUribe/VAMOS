from __future__ import annotations

import numpy as np
from collections.abc import Callable, Sequence


class TournamentSelection:
    """
    Simple tournament selection using a comparator.
    comparator(a, b) returns <0 if a better than b, >0 if b better, 0 if tie.
    """

    def __init__(
        self,
        tournament_size: int,
        comparator: Callable[[int, int], int],
        rng: np.random.Generator | None = None,
    ) -> None:
        if tournament_size <= 0:
            raise ValueError("tournament_size must be positive.")
        self.tournament_size = int(tournament_size)
        self.comparator = comparator
        self.rng = rng or np.random.default_rng()

    def __call__(self, population: Sequence[object], n_parents: int) -> np.ndarray:
        pop_size = len(population)
        if pop_size == 0:
            raise ValueError("population is empty.")
        rng = self.rng
        selected = np.empty(n_parents, dtype=int)
        for i in range(n_parents):
            contenders = rng.integers(0, pop_size, size=self.tournament_size)
            best = contenders[0]
            for idx in contenders[1:]:
                cmp = self.comparator(idx, best)
                if cmp < 0:
                    best = idx
            selected[i] = best
        return selected


class RandomSelection:
    """Uniform random parent selection."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    def __call__(self, population: Sequence[object], n_parents: int) -> np.ndarray:
        pop_size = len(population)
        if pop_size == 0:
            raise ValueError("population is empty.")
        return self.rng.integers(0, pop_size, size=n_parents)


__all__ = ["TournamentSelection", "RandomSelection"]
