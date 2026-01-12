"""Shared typing helpers for algorithm configuration objects."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, TypedDict, TypeAlias, runtime_checkable


class AlgorithmConfigDict(TypedDict, total=False):
    engine: str
    pop_size: int
    n_var: int
    n_obj: int
    result_mode: str
    archive_type: str
    archive: dict[str, Any]


AlgorithmConfigMapping: TypeAlias = Mapping[str, Any]


@runtime_checkable
class AlgorithmConfigProtocol(Protocol):
    def to_dict(self) -> AlgorithmConfigMapping: ...


AlgorithmConfigLike: TypeAlias = AlgorithmConfigProtocol | AlgorithmConfigMapping


__all__ = [
    "AlgorithmConfigDict",
    "AlgorithmConfigMapping",
    "AlgorithmConfigProtocol",
    "AlgorithmConfigLike",
]
