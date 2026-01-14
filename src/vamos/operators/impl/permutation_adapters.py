from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .permutation import PermPop, RNG


class SwapMutation:
    def __init__(self, prob: float = 0.1, **kwargs: object) -> None:
        self.prob = float(prob)

    def __call__(self, X: PermPop, rng: RNG, **kwargs: object) -> None:
        from .permutation import swap_mutation

        swap_mutation(X, self.prob, rng)


class PMXCrossover:
    def __init__(self, prob: float = 0.9, **kwargs: object) -> None:
        self.prob = float(prob)

    def __call__(self, parents: PermPop, rng: RNG, **kwargs: object) -> PermPop:
        from .permutation import pmx_crossover

        return pmx_crossover(parents, self.prob, rng)


class CycleCrossover:
    def __init__(self, prob: float = 0.9, **kwargs: object) -> None:
        self.prob = float(prob)

    def __call__(self, parents: PermPop, rng: RNG, **kwargs: object) -> PermPop:
        from .permutation import cycle_crossover

        return cycle_crossover(parents, self.prob, rng)


class PositionBasedCrossover:
    def __init__(self, prob: float = 0.9, **kwargs: object) -> None:
        self.prob = float(prob)

    def __call__(self, parents: PermPop, rng: RNG, **kwargs: object) -> PermPop:
        from .permutation import position_based_crossover

        return position_based_crossover(parents, self.prob, rng)


class EdgeRecombinationCrossover:
    def __init__(self, prob: float = 0.9, **kwargs: object) -> None:
        self.prob = float(prob)

    def __call__(self, parents: PermPop, rng: RNG, **kwargs: object) -> PermPop:
        from .permutation import edge_recombination_crossover

        return edge_recombination_crossover(parents, self.prob, rng)


class OrderCrossover:
    def __init__(self, prob: float = 0.9, **kwargs: object) -> None:
        self.prob = float(prob)

    def __call__(self, parents: PermPop, rng: RNG, **kwargs: object) -> PermPop:
        from .permutation import order_crossover

        return order_crossover(parents, self.prob, rng)


class InsertMutation:
    def __init__(self, prob: float = 0.1, **kwargs: object) -> None:
        self.prob = float(prob)

    def __call__(self, X: PermPop, rng: RNG, **kwargs: object) -> None:
        from .permutation import insert_mutation

        insert_mutation(X, self.prob, rng)


class ScrambleMutation:
    def __init__(self, prob: float = 0.1, **kwargs: object) -> None:
        self.prob = float(prob)

    def __call__(self, X: PermPop, rng: RNG, **kwargs: object) -> None:
        from .permutation import scramble_mutation

        scramble_mutation(X, self.prob, rng)


class InversionMutation:
    def __init__(self, prob: float = 0.1, **kwargs: object) -> None:
        self.prob = float(prob)

    def __call__(self, X: PermPop, rng: RNG, **kwargs: object) -> None:
        from .permutation import inversion_mutation

        inversion_mutation(X, self.prob, rng)


class DisplacementMutation:
    def __init__(self, prob: float = 0.1, **kwargs: object) -> None:
        self.prob = float(prob)

    def __call__(self, X: PermPop, rng: RNG, **kwargs: object) -> None:
        from .permutation import displacement_mutation

        displacement_mutation(X, self.prob, rng)


__all__ = [
    "SwapMutation",
    "PMXCrossover",
    "CycleCrossover",
    "PositionBasedCrossover",
    "EdgeRecombinationCrossover",
    "OrderCrossover",
    "InsertMutation",
    "ScrambleMutation",
    "InversionMutation",
    "DisplacementMutation",
]
