"""
MIC Experiment 01: representative instance selector.

This script uses exactly the instances listed in the MIC runtime table:
  - cec2009_uf1..cec2009_uf10
  - lsmop1..lsmop9
  - c1dtlz1, c1dtlz3, c2dtlz2
  - dc1dtlz1, dc1dtlz3, dc2dtlz1, dc2dtlz3
  - mw1, mw2, mw3, mw5, mw6, mw7

Selection is stratified by family (UF / LSMOP / C-DTLZ / DC-DTLZ / MW)
and seeks good diversity coverage using a centroid + farthest-point policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[3]
FAMILY_ORDER = ("UF", "LSMOP", "C-DTLZ", "DC-DTLZ", "MW")


@dataclass(frozen=True)
class ProblemInstance:
    name: str
    family: str
    n_var: int
    n_obj: int
    constrained: int
    disconnected_pf: int
    multimodal: int
    deceptive: int
    nonseparable: int
    parameter_dependency: int
    large_scale: int

    def trait_vector(self) -> np.ndarray:
        return np.array(
            [
                float(self.constrained),
                float(self.disconnected_pf),
                float(self.multimodal),
                float(self.deceptive),
                float(self.nonseparable),
                float(self.parameter_dependency),
                float(self.large_scale),
            ],
            dtype=float,
        )


def _mic_table_instances() -> list[ProblemInstance]:
    rows: list[ProblemInstance] = []

    # UF (30 vars; 2 obj for UF1-7, 3 obj for UF8-10).
    uf_objs = {f"cec2009_uf{i}": (2 if i <= 7 else 3) for i in range(1, 11)}
    for name, n_obj in uf_objs.items():
        rows.append(
            ProblemInstance(
                name=name,
                family="UF",
                n_var=30,
                n_obj=n_obj,
                constrained=0,
                disconnected_pf=1 if n_obj == 3 else 0,
                multimodal=1 if name in {"cec2009_uf3", "cec2009_uf4", "cec2009_uf5", "cec2009_uf6", "cec2009_uf7"} else 0,
                deceptive=1 if name in {"cec2009_uf9", "cec2009_uf10"} else 0,
                nonseparable=1,
                parameter_dependency=1 if name in {"cec2009_uf6", "cec2009_uf7", "cec2009_uf9", "cec2009_uf10"} else 0,
                large_scale=0,
            )
        )

    # LSMOP (MIC setup uses 100 vars, 2 objectives).
    for i in range(1, 10):
        rows.append(
            ProblemInstance(
                name=f"lsmop{i}",
                family="LSMOP",
                n_var=100,
                n_obj=2,
                constrained=0,
                disconnected_pf=0,
                multimodal=1,
                deceptive=1 if i in {5, 6, 7, 8, 9} else 0,
                nonseparable=1,
                parameter_dependency=1,
                large_scale=1,
            )
        )

    # C-DTLZ (12 vars, 2 objectives).
    rows.extend(
        [
            ProblemInstance("c1dtlz1", "C-DTLZ", 12, 2, 1, 0, 1, 0, 0, 0, 0),
            ProblemInstance("c1dtlz3", "C-DTLZ", 12, 2, 1, 0, 1, 0, 0, 0, 0),
            ProblemInstance("c2dtlz2", "C-DTLZ", 12, 2, 1, 0, 0, 0, 0, 0, 0),
        ]
    )

    # DC-DTLZ (12 vars, 2 objectives).
    rows.extend(
        [
            ProblemInstance("dc1dtlz1", "DC-DTLZ", 12, 2, 1, 0, 1, 0, 0, 0, 0),
            ProblemInstance("dc1dtlz3", "DC-DTLZ", 12, 2, 1, 0, 1, 0, 0, 0, 0),
            ProblemInstance("dc2dtlz1", "DC-DTLZ", 12, 2, 1, 0, 1, 0, 0, 0, 0),
            ProblemInstance("dc2dtlz3", "DC-DTLZ", 12, 2, 1, 0, 1, 0, 0, 0, 0),
        ]
    )

    # MW (15 vars, 2 objectives).
    rows.extend(
        [
            ProblemInstance("mw1", "MW", 15, 2, 1, 0, 0, 0, 1, 0, 0),
            ProblemInstance("mw2", "MW", 15, 2, 1, 0, 1, 0, 1, 0, 0),
            ProblemInstance("mw3", "MW", 15, 2, 1, 0, 1, 0, 1, 0, 0),
            ProblemInstance("mw5", "MW", 15, 2, 1, 0, 0, 0, 1, 0, 0),
            ProblemInstance("mw6", "MW", 15, 2, 1, 0, 0, 0, 1, 0, 0),
            ProblemInstance("mw7", "MW", 15, 2, 1, 0, 0, 0, 1, 0, 0),
        ]
    )
    return rows


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if math.isclose(lo, hi):
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _feature_matrix(instances: list[ProblemInstance]) -> np.ndarray:
    n_var = _normalize(np.array([p.n_var for p in instances], dtype=float))
    n_obj = _normalize(np.array([p.n_obj for p in instances], dtype=float))

    features: list[np.ndarray] = []
    for i, p in enumerate(instances):
        family_oh = np.array([1.0 if p.family == fam else 0.0 for fam in FAMILY_ORDER], dtype=float)
        vec = np.concatenate(
            [
                np.array([n_var[i], n_obj[i]], dtype=float),
                1.25 * family_oh,
                p.trait_vector(),
            ]
        )
        features.append(vec)
    return np.vstack(features)


def _pairwise_dist(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _resolve_k_total(total_instances: int, selection_pct: float, k_total: int | None) -> int:
    if k_total is not None:
        return int(k_total)
    if not (0.0 < selection_pct <= 100.0):
        raise ValueError("selection_pct must be in (0, 100].")
    resolved = int(round((selection_pct / 100.0) * total_instances))
    return max(1, min(total_instances, resolved))


def _allocate_quotas(instances: list[ProblemInstance], k_total: int, min_per_family: int) -> dict[str, int]:
    family_sizes = {fam: sum(1 for p in instances if p.family == fam) for fam in FAMILY_ORDER}
    n_total = sum(family_sizes.values())

    if k_total <= 0:
        raise ValueError("k_total must be positive.")
    if k_total > n_total:
        raise ValueError(f"k_total={k_total} exceeds total instances ({n_total}).")
    if min_per_family < 0:
        raise ValueError("min_per_family cannot be negative.")
    if min_per_family * len(FAMILY_ORDER) > k_total:
        raise ValueError(
            "min_per_family cannot be satisfied with current k_total. "
            f"Received: min_per_family={min_per_family}, k_total={k_total}."
        )

    target = {fam: (k_total * family_sizes[fam] / n_total) for fam in FAMILY_ORDER}
    quotas = {fam: min(family_sizes[fam], int(math.floor(target[fam]))) for fam in FAMILY_ORDER}
    floor = {fam: min(min_per_family, family_sizes[fam]) for fam in FAMILY_ORDER}

    for fam in FAMILY_ORDER:
        quotas[fam] = max(quotas[fam], floor[fam])

    def current_total() -> int:
        return sum(quotas.values())

    while current_total() > k_total:
        candidates = [fam for fam in FAMILY_ORDER if quotas[fam] > floor[fam]]
        if not candidates:
            break
        fam = min(candidates, key=lambda f: (target[f] - quotas[f], FAMILY_ORDER.index(f)))
        quotas[fam] -= 1

    while current_total() < k_total:
        candidates = [fam for fam in FAMILY_ORDER if quotas[fam] < family_sizes[fam]]
        if not candidates:
            break
        fam = max(candidates, key=lambda f: (target[f] - quotas[f], -FAMILY_ORDER.index(f)))
        quotas[fam] += 1

    return quotas


def _closest_to_centroid(indices: list[int], X: np.ndarray, names: list[str]) -> int:
    centroid = np.mean(X[indices], axis=0)
    scored: list[tuple[float, str, int]] = []
    for idx in indices:
        d = float(np.linalg.norm(X[idx] - centroid))
        scored.append((d, names[idx], idx))
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[0][2]


def _select_within_family(
    family: str,
    k: int,
    instances: list[ProblemInstance],
    X: np.ndarray,
    D: np.ndarray,
) -> list[int]:
    family_idx = [i for i, p in enumerate(instances) if p.family == family]
    if k <= 0:
        return []
    if k >= len(family_idx):
        return sorted(family_idx, key=lambda i: instances[i].name)

    names = [p.name for p in instances]
    selected = [_closest_to_centroid(family_idx, X, names)]
    selected_set = set(selected)

    while len(selected) < k:
        candidates = [i for i in family_idx if i not in selected_set]
        scored: list[tuple[float, str, int]] = []
        for cand in candidates:
            min_dist = min(float(D[cand, s]) for s in selected)
            scored.append((min_dist, instances[cand].name, cand))
        scored.sort(key=lambda t: (-t[0], t[1]))
        pick = scored[0][2]
        selected.append(pick)
        selected_set.add(pick)
    return selected


def _selection_summary(
    selected_idx: list[int],
    instances: list[ProblemInstance],
    X: np.ndarray,
    D: np.ndarray,
    quotas: dict[str, int],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    centroid_by_family = {}
    for fam in FAMILY_ORDER:
        fam_idx = [i for i, p in enumerate(instances) if p.family == fam]
        centroid_by_family[fam] = np.mean(X[fam_idx], axis=0)

    selected_set = set(selected_idx)
    rows: list[dict[str, object]] = []
    for rank, idx in enumerate(selected_idx, start=1):
        p = instances[idx]
        nearest_sel = min(float(D[idx, j]) for j in selected_idx if j != idx) if len(selected_idx) > 1 else 0.0
        dist_centroid = float(np.linalg.norm(X[idx] - centroid_by_family[p.family]))
        rows.append(
            {
                "selection_rank": rank,
                "problem": p.name,
                "family": p.family,
                "n_var": p.n_var,
                "n_obj": p.n_obj,
                "quota_family": quotas[p.family],
                "constrained": p.constrained,
                "disconnected_pf": p.disconnected_pf,
                "multimodal": p.multimodal,
                "deceptive": p.deceptive,
                "nonseparable": p.nonseparable,
                "parameter_dependency": p.parameter_dependency,
                "large_scale": p.large_scale,
                "dist_to_family_centroid": round(dist_centroid, 6),
                "dist_to_nearest_selected": round(nearest_sel, 6),
            }
        )

    nearest_d = []
    for i in range(len(instances)):
        nearest = min(float(D[i, j]) for j in selected_set)
        nearest_d.append(nearest)

    summary = {
        "coverage_mean_distance": round(float(np.mean(nearest_d)), 6),
        "coverage_max_distance": round(float(np.max(nearest_d)), 6),
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select representative instances from MIC runtime table families (UF/LSMOP/C-DTLZ/DC-DTLZ/MW)."
    )
    parser.add_argument(
        "--selection-pct",
        type=float,
        default=30.0,
        help="Default percentage of total instances to keep when --k-total is not provided (default: 40.0).",
    )
    parser.add_argument(
        "--k-total",
        type=int,
        default=None,
        help="Optional explicit total instances to select (overrides --selection-pct).",
    )
    parser.add_argument("--min-per-family", type=int, default=1, help="Minimum number of selected instances per family.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "experiments" / "mic" / "instance_selection",
        help="Output folder.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="representative_instances_mic_runtime",
        help="Output file prefix.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write files; print only.")
    args = parser.parse_args()

    instances = _mic_table_instances()
    total_instances = len(instances)
    k_total = _resolve_k_total(total_instances=total_instances, selection_pct=float(args.selection_pct), k_total=args.k_total)

    X = _feature_matrix(instances)
    D = _pairwise_dist(X)
    quotas = _allocate_quotas(instances, k_total=k_total, min_per_family=args.min_per_family)

    selected_idx: list[int] = []
    for fam in FAMILY_ORDER:
        selected_idx.extend(_select_within_family(fam, quotas[fam], instances, X, D))

    selected_idx = sorted(selected_idx, key=lambda i: (FAMILY_ORDER.index(instances[i].family), instances[i].name))
    selected_rows, coverage = _selection_summary(selected_idx, instances, X, D, quotas)

    selected_names = [instances[i].name for i in selected_idx]
    by_family = {fam: [instances[i].name for i in selected_idx if instances[i].family == fam] for fam in FAMILY_ORDER}

    report = {
        "suite": "mic_runtime_table",
        "total_instances": total_instances,
        "selection_pct": float(args.selection_pct),
        "k_total": int(k_total),
        "min_per_family": int(args.min_per_family),
        "family_sizes": {fam: sum(1 for p in instances if p.family == fam) for fam in FAMILY_ORDER},
        "family_quotas": quotas,
        "selected_instances": selected_names,
        "selected_by_family": by_family,
        "coverage": coverage,
    }

    print("=== MIC instance selector ===")
    print(f"Total instances: {total_instances}")
    print(f"Selection percentage: {args.selection_pct}%")
    print(f"Selected k: {k_total}")
    print(f"Quotas: {quotas}")
    print(f"Selection: {', '.join(selected_names)}")
    print(f"Coverage mean distance: {coverage['coverage_mean_distance']}")
    print(f"Coverage max distance: {coverage['coverage_max_distance']}")

    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pct_tag = str(args.selection_pct).replace(".", "p")
    base = f"{args.output_prefix}_p{pct_tag}_k{k_total}"
    csv_path = args.output_dir / f"{base}.csv"
    json_path = args.output_dir / f"{base}.json"

    _write_csv(csv_path, selected_rows)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()

