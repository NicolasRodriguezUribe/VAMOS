from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

from vamos.ux.studio.dm import DecisionView


def _record_payload(view: DecisionView, idx: int) -> dict:
    payload = {
        "objectives": view.front.points_F[idx].tolist(),
        "normalized_objectives": view.normalized_F[idx].tolist(),
        "problem": view.front.problem_name,
        "algorithm": view.front.algorithm_name,
    }
    if view.decoded_X:
        payload["decision_variables"] = view.decoded_X[idx]
    if view.constraints is not None:
        payload["constraints"] = view.constraints[idx].tolist()
    if view.mcdm_scores:
        payload["mcdm_scores"] = {k: float(v[idx]) for k, v in view.mcdm_scores.items() if len(v) > idx}
    return payload


def export_solutions_to_json(view: DecisionView, indices: List[int], path: Path) -> Path:
    records = [_record_payload(view, idx) for idx in indices]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return path


def export_solutions_to_csv(view: DecisionView, indices: List[int], path: Path) -> Path:
    rows = [_record_payload(view, idx) for idx in indices]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    # Flatten dictionaries for CSV
    fieldnames = set()
    for row in rows:
        for key in row.keys():
            fieldnames.add(key)
    field_list = sorted(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_list)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row.get(k)) if isinstance(row.get(k), (list, dict)) else row.get(k) for k in field_list})
    return path

