from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from canonical_runs import result_runs, run_row
from numpy.typing import NDArray


def pareto_nondominated_mask(values: NDArray[Any]) -> NDArray[np.bool_]:
    if values.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    le = values[:, None, :] <= values[None, :, :]
    lt = values[:, None, :] < values[None, :, :]
    dominated = np.any(np.all(le, axis=2) & np.any(lt, axis=2), axis=0)
    return np.asarray(~dominated, dtype=np.bool_)


def hv_2d_exact(values: np.ndarray, reference: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    ordered = values[np.argsort(values[:, 0])]
    next_x = np.concatenate([ordered[1:, 0], np.array([reference[0]])])
    widths = np.maximum(0.0, next_x - ordered[:, 0])
    heights = np.maximum(0.0, reference[1] - ordered[:, 1])
    return float(np.sum(widths * heights))


def hv_mc(
    values: np.ndarray,
    reference: np.ndarray,
    lower: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> float:
    if values.size == 0:
        return 0.0
    span = np.maximum(reference - lower, 1e-12)
    points = lower + rng.random((samples, values.shape[1])) * span
    dominated = np.zeros((samples,), dtype=bool)
    for start in range(0, values.shape[0], 512):
        candidates = values[start : start + 512]
        dominated |= np.any(np.all(points[:, None, :] >= candidates[None, :, :], axis=2), axis=1)
        if np.all(dominated):
            break
    return float(np.mean(dominated)) * float(np.prod(span))


def igd_plus(values: np.ndarray, reference_set: np.ndarray) -> float:
    if reference_set.size == 0:
        return float("nan")
    if values.size == 0:
        return float("inf")
    distances = []
    for reference in reference_set:
        difference = np.maximum(0.0, values - reference[None, :])
        distances.append(float(np.min(np.sqrt(np.sum(difference * difference, axis=1)))))
    return float(np.mean(distances))


def scan_runs(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in result_runs(results_root):
        if run.result.F is None:
            continue
        row = run_row(run, campaign=results_root.name)
        row["_objectives"] = np.asarray(run.result.F)
        rows.append(row)
    return rows


def load_ref_points(path: str | None) -> dict[str, np.ndarray]:
    if not path:
        return {}
    ref_path = Path(path).resolve()
    if not ref_path.exists():
        raise FileNotFoundError(f"--ref-points file not found: {ref_path}")
    data = json.loads(ref_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--ref-points must be a JSON object mapping problem to a reference vector.")
    output: dict[str, np.ndarray] = {}
    for key, value in data.items():
        point = np.asarray(value, dtype=float)
        if point.ndim != 1:
            raise ValueError(f"Reference point for problem '{key}' must be one-dimensional; got {point.shape}.")
        output[str(key)] = point
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-out", required=True)
    parser.add_argument("--mc-samples", type=int, default=20000)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--ref-points", default=None, help="Optional JSON mapping problem IDs to frozen reference vectors")
    parser.add_argument("--max-runs", type=int, default=0)
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    output = Path(args.out).resolve()
    sample_output = Path(args.sample_out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_output.parent.mkdir(parents=True, exist_ok=True)

    runs = scan_runs(results_root)
    if args.max_runs:
        runs = runs[: args.max_runs]
    if not runs:
        print("No canonical runs found under:", results_root)
        return 2

    frozen_ref = load_ref_points(args.ref_points)
    by_problem: dict[str, list[np.ndarray]] = {}
    for row in runs:
        by_problem.setdefault(str(row["problem"]), []).append(row["_objectives"])

    problem_stats: dict[str, dict[str, Any]] = {}
    reference_sets: dict[str, np.ndarray] = {}
    hv_references: dict[str, np.ndarray] = {}
    hv_lowers: dict[str, np.ndarray] = {}
    for problem, fronts in by_problem.items():
        combined = np.vstack(fronts)
        lower = np.min(combined, axis=0)
        maximum = np.max(combined, axis=0)
        reference = frozen_ref.get(problem, maximum + 0.05 * np.maximum(maximum - lower, 1e-12) + 1e-9)
        if reference.shape[0] != combined.shape[1]:
            raise ValueError(f"Reference point dimension mismatch for '{problem}'.")
        reference_set = np.unique(combined[pareto_nondominated_mask(combined)], axis=0)
        hv_lowers[problem] = lower
        hv_references[problem] = reference
        reference_sets[problem] = reference_set
        problem_stats[problem] = {
            "n_obj": int(combined.shape[1]),
            "ref": reference.tolist(),
            "ref_source": "frozen" if problem in frozen_ref else "observed_max_margin",
            "lo": lower.tolist(),
            "refset_size": int(reference_set.shape[0]),
        }

    rng = np.random.default_rng(args.rng_seed)
    output_rows: list[dict[str, Any]] = []
    for source in runs:
        row = {key: value for key, value in source.items() if not key.startswith("_")}
        problem = str(row["problem"])
        values = source["_objectives"]
        nondominated = values[pareto_nondominated_mask(values)]
        reference = hv_references[problem]
        if nondominated.shape[1] == 2:
            hypervolume = hv_2d_exact(nondominated, reference)
        else:
            hypervolume = hv_mc(nondominated, reference, hv_lowers[problem], args.mc_samples, rng)
        row.update(
            {
                "nd_size": int(nondominated.shape[0]),
                "hv_ref": json.dumps(reference.tolist()),
                "hv_final": hypervolume,
                "igd_plus": igd_plus(nondominated, reference_sets[problem]),
                "refset_size": int(reference_sets[problem].shape[0]),
            }
        )
        output_rows.append(row)

    columns = sorted({key for row in output_rows for key in row})
    for path, rows in ((output, output_rows), (sample_output, output_rows[:60])):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    stats_path = output.with_suffix(".problem_stats.json")
    stats_path.write_text(json.dumps(problem_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote:", output)
    print("Wrote derived sample:", sample_output)
    print("Wrote problem statistics:", stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
