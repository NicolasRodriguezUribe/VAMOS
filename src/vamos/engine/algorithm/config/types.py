"""Shared typing helpers for algorithm configuration objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeAlias, runtime_checkable


AlgorithmConfigMapping: TypeAlias = Mapping[str, object]


@runtime_checkable
class AlgorithmConfigProtocol(Protocol):
    def to_dict(self) -> AlgorithmConfigMapping: ...


__all__ = [
    "AlgorithmConfigMapping",
    "AlgorithmConfigProtocol",
]
