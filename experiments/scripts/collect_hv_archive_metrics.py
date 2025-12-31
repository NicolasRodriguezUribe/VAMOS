from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def pareto_nondominated_mask(F: np.ndarray) -> np.ndarray:
    n = F.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)
    le = (F[:, None, :] <= F[None, :, :])
    lt = (F[:, None, :] < F[None, :, :])
    dom = np.all(le, axis=2) & np.any(lt, axis=2)
    dominated = np.any(dom, axis=0)
    return ~dominated


def hv_2d_exact(F: np.ndarray, ref: np.ndarray) -> float:
    if F.size == 0:
        return 0.0
    idx = np.argsort(F[:, 0])
    P = F[idx]
    xs = P[:, 0]
    ys = P[:, 1]
    next_x = np.concatenate([xs[1:], np.array([ref[0]])])
    widths = np.maximum(0.0, next_x - xs)
    heights = np.maximum(0.0, ref[1] - ys)
    return float(np.sum(widths * heights))


def hv_mc(F: np.ndarray, ref: np.ndarray, lo: np.ndarray, samples: int, rng: np.random.Generator) -> float:
    """
    Monte Carlo HV estimate over box [lo, ref] for minimization:
    a sample point u is dominated by set if exists s in F with u >= s in all dims.
    HV = P(dominated) * volume(box).
    """
    if F.size == 0:
        return 0.0
    m = F.shape[1]
    ref = np.asarray(ref, dtype=float)
    lo = np.asarray(lo, dtype=float)
    span = np.maximum(ref - lo, 1e-12)
    U = rng.random((samples, m))
    pts = lo + U * span

    dom_any = np.zeros((samples,), dtype=bool)
    chunk = 512
    for i in range(0, F.shape[0], chunk):
        S = F[i : i + chunk]
        dom = np.any(np.all(pts[:, None, :] >= S[None, :, :], axis=2), axis=1)
        dom_any |= dom
        if np.all(dom_any):
            break
    frac = float(np.mean(dom_any))
    vol = float(np.prod(span))
    return frac * vol


def igd_plus(F: np.ndarray, R: np.ndarray) -> float:
    """
    IGD+ (minimization): for each reference point r in R,
    distance to solution set S is min_s sqrt(sum_i max(0, s_i - r_i)^2).
    """
    if R.size == 0:
        return float("nan")
    if F.size == 0:
        return float("inf")
    out = []
    for r in R:
        dif = np.maximum(0.0, F - r[None, :])
        d = np.sqrt(np.sum(dif * dif, axis=1))
        out.append(float(np.min(d)))
    return float(np.mean(out))


def read_csv_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(str(path), delimiter=",")


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def infer_suite_from_problem(problem_key: str) -> str:
    import re

    key = str(problem_key).lower()
    match = re.match(r"^([a-z]+)", key)
    prefix = match.group(1) if match else "unknown"
    return prefix.upper()


def _normalize_problem(meta_problem: Any) -> str:
    if isinstance(meta_problem, dict):
        return str(meta_problem.get("key") or meta_problem.get("label") or "unknown")
    return str(meta_problem or "unknown")


def scan_runs(results_root: Path) -> List[Dict[str, Any]]:
    runs = []
    for md_path in results_root.rglob("metadata.json"):
        try:
            meta = json.loads(md_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        seed_dir = md_path.parent
        fun = seed_dir / "FUN.csv"
        time_txt = seed_dir / "time.txt"
        hv_trace = seed_dir / "hv_trace.csv"
        ar_stats = seed_dir / "archive_stats.csv"
        if not fun.exists():
            continue

        rel = seed_dir.relative_to(results_root)
        parts = rel.parts
        variant = parts[0] if len(parts) >= 1 else "unknown"

        algo = meta.get("algorithm", meta.get("config", {}).get("algorithm", "unknown"))
        engine = meta.get("backend", meta.get("config", {}).get("engine", "unknown"))
        problem = _normalize_problem(meta.get("problem", meta.get("config", {}).get("problem", "unknown")))
        suite = infer_suite_from_problem(problem)

        seed = meta.get("seed", None)
        max_evals = meta.get("max_evaluations", meta.get("config", {}).get("max_evaluations", None))
        pop = meta.get("population_size", meta.get("config", {}).get("population_size", None))

        stopping = meta.get("stopping", {}) if isinstance(meta.get("stopping", {}), dict) else {}
        stop_enabled = bool(stopping.get("enabled", False))
        stop_triggered = bool(stopping.get("triggered", False))
        evals_stop = stopping.get("evals_stop", None)

        archive = meta.get("archive", {}) if isinstance(meta.get("archive", {}), dict) else {}
        arch_enabled = bool(archive.get("enabled", False))
        arch_final = archive.get("final_size", None)

        runtime_s = None
        if time_txt.exists():
            try:
                runtime_ms = float(time_txt.read_text(encoding="utf-8").strip())
                runtime_s = runtime_ms / 1000.0
            except Exception:
                runtime_s = None
        if runtime_s is None:
            metrics = meta.get("metrics", {}) if isinstance(meta.get("metrics", {}), dict) else {}
            runtime_s = safe_float(metrics.get("runtime_seconds"))

        runs.append(
            {
                "seed_dir": str(seed_dir),
                "variant": variant,
                "suite": suite,
                "algorithm": str(algo),
                "engine": str(engine),
                "problem": str(problem),
                "seed": seed,
                "max_evaluations": max_evals,
                "population_size": pop,
                "FUN": str(fun),
                "runtime_s": runtime_s,
                "hv_trace": str(hv_trace) if hv_trace.exists() else None,
                "archive_stats": str(ar_stats) if ar_stats.exists() else None,
                "stop_enabled": stop_enabled,
                "stop_triggered": stop_triggered,
                "evals_stop": evals_stop,
                "arch_enabled": arch_enabled,
                "arch_final_size": arch_final,
            }
        )
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-out", required=True)
    ap.add_argument("--mc-samples", type=int, default=20000)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--max-runs", type=int, default=0)
    args = ap.parse_args()

    results_root = Path(args.results_root).resolve()
    out_path = Path(args.out).resolve()
    sample_path = Path(args.sample_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    runs = scan_runs(results_root)
    if args.max_runs and len(runs) > args.max_runs:
        runs = runs[: args.max_runs]

    print("results_root:", results_root)
    print("runs_found:", len(runs))
    if not runs:
        return 2

    by_problem: Dict[str, List[np.ndarray]] = {}
    for r in runs:
        F = read_csv_matrix(Path(r["FUN"]))
        if F.ndim == 1:
            F = F.reshape(1, -1)
        by_problem.setdefault(r["problem"], []).append(F)

    problem_stats: Dict[str, Dict[str, Any]] = {}
    refset_ultimate: Dict[str, np.ndarray] = {}
    hv_ref: Dict[str, np.ndarray] = {}
    hv_lo: Dict[str, np.ndarray] = {}

    for prob, Fs in by_problem.items():
        allF = np.vstack(Fs)
        lo = np.min(allF, axis=0)
        mx = np.max(allF, axis=0)
        margin = 0.05 * np.maximum(mx - lo, 1e-12)
        ref = mx + margin + 1e-9
        hv_ref[prob] = ref
        hv_lo[prob] = lo

        nd_mask = pareto_nondominated_mask(allF)
        R = allF[nd_mask]
        R = np.unique(R, axis=0)
        refset_ultimate[prob] = R

        problem_stats[prob] = {
            "n_obj": int(allF.shape[1]),
            "ref": ref.tolist(),
            "lo": lo.tolist(),
            "refset_size": int(R.shape[0]),
        }

    rng = np.random.default_rng(int(args.rng_seed))

    rows: List[Dict[str, Any]] = []
    for r in runs:
        prob = r["problem"]
        F = read_csv_matrix(Path(r["FUN"]))
        if F.ndim == 1:
            F = F.reshape(1, -1)
        nd = F[pareto_nondominated_mask(F)]
        ref = hv_ref[prob]
        lo = hv_lo[prob]
        m = nd.shape[1]

        if m == 2:
            hv_val = hv_2d_exact(nd, ref=ref)
        else:
            hv_val = hv_mc(nd, ref=ref, lo=lo, samples=int(args.mc_samples), rng=rng)

        igd = igd_plus(nd, refset_ultimate[prob])

        row = dict(r)
        row.update(
            {
                "n_obj": int(m),
                "nd_size": int(nd.shape[0]),
                "hv_ref": json.dumps(ref.tolist()),
                "hv_final": float(hv_val),
                "igd_plus": float(igd),
                "refset_size": int(refset_ultimate[prob].shape[0]),
            }
        )
        rows.append(row)

    import csv

    cols = sorted({k for row in rows for k in row.keys()})
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    sample = rows[: min(60, len(rows))]
    with sample_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in sample:
            w.writerow(row)

    stats_path = out_path.with_suffix(".problem_stats.json")
    stats_path.write_text(json.dumps(problem_stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Wrote:", out_path)
    print("Wrote sample:", sample_path)
    print("Wrote problem_stats:", stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
