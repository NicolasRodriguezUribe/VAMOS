from __future__ import annotations

from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np


@runtime_checkable
class VariationOperator(Protocol):
    """
    Protocol for any variation operator (crossover, mutation, repair).
    """

    def __call__(self, X: np.ndarray, rng: np.random.Generator, **kwargs: Any) -> np.ndarray:
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


RealCrossoverName: TypeAlias = Literal[
    "sbx", "blx_alpha", "blx_alpha_beta", "arithmetic", "whole_arithmetic",
    "laplace", "fuzzy", "pcx", "undx", "simplex", "de",
]
RealMutationName: TypeAlias = Literal[
    "pm",
    "polynomial",
    "non_uniform",
    "gaussian",
    "uniform_reset",
    "cauchy",
    "uniform",
    "linked_polynomial",
    "levy_flight",
    "power_law",
]
RealRepairName: TypeAlias = Literal["clip", "clamp", "reflect", "random", "resample", "round"]

BinaryCrossoverName: TypeAlias = Literal["one_point", "single_point", "1point", "spx", "two_point", "2point", "uniform", "hux"]
BinaryMutationName: TypeAlias = Literal["bitflip", "bit_flip"]

IntegerCrossoverName: TypeAlias = Literal["uniform", "blend", "arithmetic", "sbx"]
IntegerMutationName: TypeAlias = Literal["reset", "random_reset", "creep", "pm", "polynomial"]

PermutationCrossoverName: TypeAlias = Literal[
    "ox",
    "order",
    "oxd",
    "pmx",
    "cycle",
    "cx",
    "position",
    "position_based",
    "pos",
    "edge",
    "edge_recombination",
    "erx",
]
PermutationMutationName: TypeAlias = Literal["swap", "insert", "scramble", "inversion", "displacement"]

MixedCrossoverName: TypeAlias = Literal["mixed", "uniform"]
MixedMutationName: TypeAlias = Literal["mixed", "gaussian"]

CrossoverName: TypeAlias = RealCrossoverName | BinaryCrossoverName | IntegerCrossoverName | PermutationCrossoverName | MixedCrossoverName
MutationName: TypeAlias = RealMutationName | BinaryMutationName | IntegerMutationName | PermutationMutationName | MixedMutationName
RepairName: TypeAlias = RealRepairName
OperatorName: TypeAlias = CrossoverName | MutationName | RepairName


@runtime_checkable
class VariationWorkspaceProtocol(Protocol):
    """Workspace contract for reusable NumPy buffers (see `VariationWorkspace`)."""

    def request(self, key: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray: ...


@runtime_checkable
class CrossoverOperator(Protocol):
    """Vectorized crossover operator: parents -> offspring."""

    def __call__(self, parents: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...


@runtime_checkable
class MutationOperator(Protocol):
    """Vectorized mutation operator: offspring -> offspring (may mutate in-place)."""

    def __call__(self, offspring: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...


@runtime_checkable
class RepairOperator(Protocol):
    """Vectorized repair operator: clamp/reflect/resample etc."""

    def __call__(self, X: np.ndarray, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...


__all__ = [
    "BinaryCrossoverName",
    "BinaryMutationName",
    "CrossoverName",
    "IntegerCrossoverName",
    "IntegerMutationName",
    "MixedCrossoverName",
    "MixedMutationName",
    "MutationName",
    "OperatorName",
    "PermutationCrossoverName",
    "PermutationMutationName",
    "RealCrossoverName",
    "RealMutationName",
    "RealRepairName",
    "RepairName",
    "CrossoverOperator",
    "MutationOperator",
    "RepairOperator",
    "VariationOperator",
    "VariationWorkspaceProtocol",
]
