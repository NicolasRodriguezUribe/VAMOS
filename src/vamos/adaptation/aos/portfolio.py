"""
Operator portfolio primitives for AOS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class OperatorArm:
    """
    A single operator arm in the portfolio.
    """

    op_id: str
    name: str


class OperatorPortfolio:
    """
    Registry-style portfolio with stable arm ordering.
    """

    def __init__(self, arms: Sequence[OperatorArm]):
        if not arms:
            raise ValueError("OperatorPortfolio requires at least one arm.")
        self._arms = list(arms)
        self._index = {}
        for idx, arm in enumerate(self._arms):
            if not arm.op_id:
                raise ValueError("OperatorArm.op_id must be non-empty.")
            if arm.op_id in self._index:
                raise ValueError(f"Duplicate operator id '{arm.op_id}'.")
            self._index[arm.op_id] = idx

    @classmethod
    def from_ids(cls, op_ids: Iterable[str]) -> "OperatorPortfolio":
        arms = [OperatorArm(op_id=op_id, name=op_id) for op_id in op_ids]
        return cls(arms)

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]]) -> "OperatorPortfolio":
        arms = [OperatorArm(op_id=op_id, name=name) for op_id, name in pairs]
        return cls(arms)

    def by_id(self, op_id: str) -> OperatorArm:
        return self._arms[self._index[op_id]]

    def index_of(self, op_id: str) -> int:
        return self._index[op_id]

    def ids(self) -> list[str]:
        return [arm.op_id for arm in self._arms]

    def names(self) -> list[str]:
        return [arm.name for arm in self._arms]

    def __len__(self) -> int:
        return len(self._arms)

    def __iter__(self):
        return iter(self._arms)

    def __getitem__(self, index: int) -> OperatorArm:
        return self._arms[index]


__all__ = ["OperatorArm", "OperatorPortfolio"]
