from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Dict

import numpy as np


IndividualID = int


@dataclass
class GenealogyRecord:
    individual_id: IndividualID
    generation: int
    parents: List[IndividualID]
    operator_name: Optional[str]
    algorithm_name: Optional[str]
    fitness: Optional[np.ndarray] = None
    is_final_front: bool = False


class GenealogyTracker(Protocol):
    records: Dict[IndividualID, GenealogyRecord]
    next_id: int

    def new_individual(
        self,
        generation: int,
        parents: List[IndividualID],
        operator_name: str | None,
        algorithm_name: str | None,
        fitness: Optional[np.ndarray] = None,
    ) -> IndividualID:
        ...

    def mark_final_front(self, ids: List[IndividualID]) -> None:
        ...


@dataclass
class DefaultGenealogyTracker:
    records: Dict[IndividualID, GenealogyRecord] = field(default_factory=dict)
    next_id: int = 0

    def new_individual(
        self,
        generation: int,
        parents: List[IndividualID],
        operator_name: str | None,
        algorithm_name: str | None,
        fitness: Optional[np.ndarray] = None,
    ) -> IndividualID:
        idx = self.next_id
        self.next_id += 1
        self.records[idx] = GenealogyRecord(
            individual_id=idx,
            generation=generation,
            parents=list(parents),
            operator_name=operator_name,
            algorithm_name=algorithm_name,
            fitness=fitness.copy() if fitness is not None else None,
        )
        return idx

    def mark_final_front(self, ids: List[IndividualID]) -> None:
        for i in ids:
            if i in self.records:
                self.records[i].is_final_front = True


class NoOpGenealogyTracker:
    records: Dict[IndividualID, GenealogyRecord] = {}
    next_id: int = 0

    def new_individual(
        self,
        generation: int,
        parents: List[IndividualID],
        operator_name: str | None,
        algorithm_name: str | None,
        fitness: Optional[np.ndarray] = None,
    ) -> IndividualID:
        return -1

    def mark_final_front(self, ids: List[IndividualID]) -> None:
        return None


def get_lineage(tracker: GenealogyTracker, individual_id: IndividualID) -> List[GenealogyRecord]:
    lineage: List[GenealogyRecord] = []
    stack = [individual_id]
    visited = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        rec = tracker.records.get(current)
        if rec is None:
            continue
        lineage.append(rec)
        stack.extend(rec.parents)
    return lineage


__all__ = [
    "IndividualID",
    "GenealogyRecord",
    "GenealogyTracker",
    "DefaultGenealogyTracker",
    "NoOpGenealogyTracker",
    "get_lineage",
]
