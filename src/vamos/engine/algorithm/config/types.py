"""Shared typing helpers for algorithm configuration objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypedDict, TypeAlias, runtime_checkable


class AlgorithmConfigDict(TypedDict, total=False):
    pop_size: int
    n_var: int
    n_obj: int
    result_mode: str
    archive_type: str
    archive: dict[str, object]


AlgorithmConfigMapping: TypeAlias = Mapping[str, object]


@runtime_checkable
class AlgorithmConfigProtocol(Protocol):
    def to_dict(self) -> AlgorithmConfigMapping: ...


__all__ = [
    "AlgorithmConfigDict",
    "AlgorithmConfigMapping",
    "AlgorithmConfigProtocol",
]
