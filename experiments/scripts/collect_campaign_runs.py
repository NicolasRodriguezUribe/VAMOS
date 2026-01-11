from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def try_parse_float(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?", s)
    if not m:
        return None
    t = m.group(0).replace(",", ".")
    try:
        return float(t)
    except Exception:
        return None


def flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flatten(key, v, out)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        out[prefix] = obj
    else:
        if isinstance(obj, list):
            out[prefix] = f"<list len={len(obj)}>"
        else:
            out[prefix] = f"<{type(obj).__name__}>"


def is_nan(x: float) -> bool:
    return x != x


def read_csv_matrix(p: Path) -> Tuple[int, int, List[List[float]]]:
    rows: List[List[float]] = []
    import csv

    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            if len(r) == 1 and (" " in r[0] or "\t" in r[0]):
                r = re.split(r"[,\s]+", r[0].strip())
            vals: List[float] = []
            for x in r:
                x = x.strip()
                if x == "":
                    continue
                try:
                    vals.append(float(x))
                except Exception:
                    vals.append(float("nan"))
            if vals:
                rows.append(vals)
    nrows = len(rows)
    ncols = max((len(r) for r in rows), default=0)
    for r in rows:
        if len(r) < ncols:
            r.extend([float("nan")] * (ncols - len(r)))
    return nrows, ncols, rows


def col_min_max(mat: List[List[float]], j: int) -> Tuple[float | None, float | None]:
    vals = [r[j] for r in mat if j < len(r) and not is_nan(r[j])]
    if not vals:
        return None, None
    return min(vals), max(vals)


def infer_from_seed_dir(sd: Path) -> Dict[str, str]:
    # expected: .../<suite>/<algo>/<engine>/seed_<k>
    return {
        "suite": sd.parents[2].name if len(sd.parents) >= 3 else "",
        "algorithm": sd.parent.parent.name if len(sd.parents) >= 2 else "",
        "engine": sd.parent.name,
    }


def row_from_seed_dir(sd: Path, campaign: str) -> Dict[str, Any]:
    r: Dict[str, Any] = {}
    r["run_path"] = sd.as_posix()
    r["campaign"] = campaign

    inf = infer_from_seed_dir(sd)
    r.update(inf)

    meta_p = sd / "metadata.json"
    if meta_p.exists():
        meta = read_json(meta_p)
        for k in [
            "algorithm",
            "problem",
            "seed",
            "max_evaluations",
            "population_size",
            "vamos_version",
            "git_revision",
            "timestamp",
            "backend",
        ]:
            if k in meta:
                r[k] = meta[k]
        # backend_info + metrics flattened
        if "backend_info" in meta and isinstance(meta["backend_info"], dict):
            tmp: Dict[str, Any] = {}
            flatten("backend_info", meta["backend_info"], tmp)
            r.update(tmp)
        if "metrics" in meta and isinstance(meta["metrics"], dict):
            tmp = {}
            flatten("metrics", meta["metrics"], tmp)
            r.update(tmp)
        # config keys snapshot
        cfg = meta.get("config", {})
        if isinstance(cfg, dict):
            r["config_keys"] = ",".join(sorted(cfg.keys()))
    else:
        r["metadata_missing"] = True

    # runtime
    time_p = sd / "time.txt"
    r["runtime_seconds"] = try_parse_float(time_p.read_text(encoding="utf-8", errors="replace")) if time_p.exists() else None

    # FUN/X
    fun_p = sd / "FUN.csv"
    x_p = sd / "X.csv"
    if fun_p.exists():
        nrows, ncols, mat = read_csv_matrix(fun_p)
        r["front_size"] = nrows
        r["fun_ncols"] = ncols
        for j in range(ncols):
            mn, mx = col_min_max(mat, j)
            r[f"obj{j}_min"] = mn
            r[f"obj{j}_max"] = mx
    if x_p.exists():
        nrows_x, ncols_x, _ = read_csv_matrix(x_p)
        r["x_nrows"] = nrows_x
        r["x_ncols"] = ncols_x

    return r


def read_index(index_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True, help="campaign name (also output csv name)")
    ap.add_argument("--results-root", required=True, help="results/<campaign> directory")
    ap.add_argument("--out", default=None, help="artifacts/tidy/<campaign>.csv (default)")
    ap.add_argument("--sample-out", default=None, help="experiments/sample_outputs/<campaign>_sample.csv (default)")
    ap.add_argument("--sample-n", type=int, default=12)
    args = ap.parse_args()

    repo = Path.cwd()
    results_root = (repo / args.results_root).resolve()
    if not results_root.exists():
        print("ERROR: results root not found:", results_root)
        return 2

    out = (repo / (args.out or f"artifacts/tidy/{args.campaign}.csv")).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    sample_out = (repo / (args.sample_out or f"experiments/sample_outputs/{args.campaign}_sample.csv")).resolve()
    sample_out.parent.mkdir(parents=True, exist_ok=True)

    index_path = results_root / "runs_index.jsonl"
    seed_dirs: List[Path] = []

    if index_path.exists():
        idx = read_index(index_path)
        # Prefer seed_dir from index for robustness
        for rec in idx:
            sd_rel = rec.get("seed_dir")
            if sd_rel:
                sd = (repo / sd_rel).resolve()
                if sd.exists():
                    seed_dirs.append(sd)
        # Deduplicate
        seed_dirs = sorted(list(dict.fromkeys(seed_dirs)))
        if seed_dirs:
            print("Using index:", index_path.relative_to(repo), "runs:", len(seed_dirs))
        else:
            print("Index present but no valid seed dirs; scanning results instead.")
            seed_dirs = sorted([p for p in results_root.rglob("seed_*") if p.is_dir() and (p / "metadata.json").exists()])
            print("Scan found seed dirs:", len(seed_dirs))
    else:
        seed_dirs = sorted([p for p in results_root.rglob("seed_*") if p.is_dir() and (p / "metadata.json").exists()])
        print("Index not found; scanning seed dirs:", len(seed_dirs))

    if not seed_dirs:
        print("ERROR: no runs found.")
        return 3

    rows: List[Dict[str, Any]] = []
    keys: set[str] = set()
    for sd in seed_dirs:
        r = row_from_seed_dir(sd, args.campaign)
        rows.append(r)
        keys |= set(r.keys())

    core = [
        "run_path",
        "campaign",
        "suite",
        "algorithm",
        "engine",
        "problem",
        "seed",
        "max_evaluations",
        "population_size",
        "runtime_seconds",
        "front_size",
        "fun_ncols",
        "x_nrows",
        "x_ncols",
        "git_revision",
        "timestamp",
        "vamos_version",
        "backend",
        "config_keys",
    ]
    cols = core + sorted([k for k in keys if k not in core])

    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write sample (first N rows)
    with sample_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows[: args.sample_n]:
            w.writerow(r)

    print("Wrote:", out.relative_to(repo), "rows:", len(rows), "cols:", len(cols))
    print("Wrote sample:", sample_out.relative_to(repo), "rows:", min(len(rows), args.sample_n))

    # Preview
    lines = out.read_text(encoding="utf-8").splitlines()
    print("\nPreview:")
    print("\n".join(lines[: min(6, len(lines))]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
