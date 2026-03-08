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
    "sbx",
    "blx_alpha",
    "blx_alpha_beta",
    "arithmetic",
    "whole_arithmetic",
    "laplace",
    "fuzzy",
    "pcx",
    "undx",
    "simplex",
    "de",
]
RealIntensificationName: TypeAlias = Literal["pave", "directional"]
RealMutationName: TypeAlias = Literal[
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
RealRepairName: TypeAlias = Literal[
    "clip", "clamp", "reflect", "random", "resample", "round", "wrap", "wrapping", "midpoint", "midpoint_base", "gradient"
]

BinaryCrossoverName: TypeAlias = Literal["one_point", "single_point", "1point", "spx", "two_point", "2point", "uniform", "hux"]
BinaryMutationName: TypeAlias = Literal["bitflip", "bit_flip", "segment_inversion"]

IntegerCrossoverName: TypeAlias = Literal["uniform", "blend", "arithmetic", "sbx"]
IntegerMutationName: TypeAlias = Literal["reset", "random_reset", "creep", "polynomial", "gaussian", "boundary"]

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
    "aex",
    "alternating_edges",
]
PermutationMutationName: TypeAlias = Literal["swap", "insert", "scramble", "inversion", "displacement", "two_opt"]

MixedCrossoverName: TypeAlias = Literal["mixed", "uniform"]
MixedMutationName: TypeAlias = Literal["mixed", "gaussian"]

CrossoverName: TypeAlias = RealCrossoverName | BinaryCrossoverName | IntegerCrossoverName | PermutationCrossoverName | MixedCrossoverName
IntensificationName: TypeAlias = RealIntensificationName
MutationName: TypeAlias = RealMutationName | BinaryMutationName | IntegerMutationName | PermutationMutationName | MixedMutationName
RepairName: TypeAlias = RealRepairName
RepairConfigValue: TypeAlias = tuple[RepairName, dict[str, Any]] | Literal["auto"]
OperatorName: TypeAlias = CrossoverName | IntensificationName | MutationName | RepairName


@runtime_checkable
class VariationWorkspaceProtocol(Protocol):
    """Workspace contract for reusable NumPy buffers (see `VariationWorkspace`)."""

    population: np.ndarray | None
    objectives: np.ndarray | None
    decision_vectors: np.ndarray | None

    def request(self, key: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray: ...

    def bind_population(self, X: np.ndarray, F: np.ndarray | None = None) -> None: ...

    def clear_population(self) -> None: ...


@runtime_checkable
class CrossoverOperator(Protocol):
    """Vectorized crossover operator: parents -> offspring."""

    def __call__(self, parents: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...


@runtime_checkable
class MutationOperator(Protocol):
    """Vectorized mutation operator: offspring -> offspring (may mutate in-place)."""

    def __call__(self, offspring: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...


@runtime_checkable
class IntensificationOperator(Protocol):
    """Vectorized intensification operator: offspring -> offspring."""

    def __call__(
        self,
        offspring: np.ndarray,
        rng: np.random.Generator,
        *,
        parents: np.ndarray | None = None,
    ) -> np.ndarray: ...


@runtime_checkable
class RepairOperator(Protocol):
    """Vectorized repair operator: clamp/reflect/resample etc."""

    def __call__(self, X: np.ndarray, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...


__all__ = [
    "BinaryCrossoverName",
    "BinaryMutationName",
    "CrossoverName",
    "IntegerCrossoverName",
    "IntensificationName",
    "IntensificationOperator",
    "IntegerMutationName",
    "MixedCrossoverName",
    "MixedMutationName",
    "MutationName",
    "OperatorName",
    "PermutationCrossoverName",
    "PermutationMutationName",
    "RealCrossoverName",
    "RealIntensificationName",
    "RealMutationName",
    "RealRepairName",
    "RepairName",
    "RepairConfigValue",
    "CrossoverOperator",
    "MutationOperator",
    "RepairOperator",
    "VariationOperator",
    "VariationWorkspaceProtocol",
]
