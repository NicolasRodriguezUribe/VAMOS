from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


IndividualID = int


@dataclass
class GenealogyRecord:
    individual_id: IndividualID
    generation: int
    parents: list[IndividualID]
    operator_name: str | None
    algorithm_name: str | None
    fitness: np.ndarray | None = None
    is_final_front: bool = False


@dataclass
class GenealogyTracker:
    records: dict[IndividualID, GenealogyRecord] = field(default_factory=dict)
    next_id: int = 0

    def new_individual(
        self,
        generation: int,
        parents: list[IndividualID],
        operator_name: str | None,
        algorithm_name: str | None,
        fitness: np.ndarray | None = None,
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

    def mark_final_front(self, ids: list[IndividualID]) -> None:
        for i in ids:
            if i in self.records:
                self.records[i].is_final_front = True


def get_lineage(tracker: GenealogyTracker, individual_id: IndividualID) -> list[GenealogyRecord]:
    lineage: list[GenealogyRecord] = []
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


def compute_operator_success_stats(tracker: GenealogyTracker) -> Any:
    import pandas as pd

    total: dict[str, int] = {}
    final: dict[str, int] = {}
    final_ids = [rid for rid, rec in tracker.records.items() if rec.is_final_front]
    final_ancestors = set()
    for fid in final_ids:
        for rec in get_lineage(tracker, fid):
            final_ancestors.add(rec.individual_id)
    for rec in tracker.records.values():
        if rec.operator_name is None:
            continue
        total[rec.operator_name] = total.get(rec.operator_name, 0) + 1
        if rec.individual_id in final_ancestors:
            final[rec.operator_name] = final.get(rec.operator_name, 0) + 1
    rows = []
    for op, cnt in total.items():
        rows.append(
            {
                "operator": op,
                "total_uses": cnt,
                "uses_in_final_lineages": final.get(op, 0),
                "ratio": (final.get(op, 0) / cnt) if cnt else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_generation_contributions(tracker: GenealogyTracker) -> Any:
    import pandas as pd

    final_ids = [rid for rid, rec in tracker.records.items() if rec.is_final_front]
    final_ancestors = set()
    for fid in final_ids:
        for rec in get_lineage(tracker, fid):
            final_ancestors.add(rec.individual_id)
    gen_totals: dict[int, int] = {}
    gen_final: dict[int, int] = {}
    for rec in tracker.records.values():
        gen_totals[rec.generation] = gen_totals.get(rec.generation, 0) + 1
        if rec.individual_id in final_ancestors:
            gen_final[rec.generation] = gen_final.get(rec.generation, 0) + 1
    rows = []
    for gen, tot in sorted(gen_totals.items()):
        rows.append(
            {
                "generation": gen,
                "total": tot,
                "final_lineage": gen_final.get(gen, 0),
                "ratio": (gen_final.get(gen, 0) / tot) if tot else 0.0,
            }
        )
    return pd.DataFrame(rows)
