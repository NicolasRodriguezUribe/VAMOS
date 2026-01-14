from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Iterable


def read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def try_parse_float(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    # accept "1.23", "1,23", "time: 1.23"
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?", s)
    if not m:
        return None
    t = m.group(0).replace(",", ".")
    try:
        return float(t)
    except Exception:
        return None


def flatten(prefix: str, obj: Any, out: dict[str, Any]) -> None:
    """Flatten nested dicts into dot-keys; ignore huge lists/structures."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flatten(key, v, out)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        out[prefix] = obj
    else:
        # For lists/tuples or custom types: store a short descriptor only.
        if isinstance(obj, list):
            out[prefix] = f"<list len={len(obj)}>"
        else:
            out[prefix] = f"<{type(obj).__name__}>"


def safe_get(d: dict[str, Any], key: str) -> Any:
    return d.get(key, None)


def find_first_key_recursively(obj: Any, wanted: str) -> Any:
    """Return first occurrence of key in nested dicts."""
    if isinstance(obj, dict):
        if wanted in obj:
            return obj[wanted]
        for _, v in obj.items():
            got = find_first_key_recursively(v, wanted)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = find_first_key_recursively(v, wanted)
            if got is not None:
                return got
    return None


def read_csv_matrix(p: Path) -> tuple[int, int, list[list[float]]]:
    rows: list[list[float]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            # Some outputs may be space-separated; handle single-cell with spaces.
            if len(r) == 1 and (" " in r[0] or "\t" in r[0]):
                r = re.split(r"[,\s]+", r[0].strip())
            vals: list[float] = []
            for x in r:
                x = x.strip()
                if x == "":
                    continue
                try:
                    vals.append(float(x))
                except Exception:
                    # keep non-numeric as NaN
                    vals.append(float("nan"))
            if vals:
                rows.append(vals)
    nrows = len(rows)
    ncols = max((len(r) for r in rows), default=0)
    # pad ragged rows with NaN
    for r in rows:
        if len(r) < ncols:
            r.extend([float("nan")] * (ncols - len(r)))
    return nrows, ncols, rows


def col_min_max(mat: list[list[float]], j: int) -> tuple[float | None, float | None]:
    vals = [r[j] for r in mat if j < len(r) and not math.isnan(r[j])]
    if not vals:
        return None, None
    return min(vals), max(vals)


def infer_from_path(seed_dir: Path) -> dict[str, str]:
    # expected: .../<suite>/<algorithm>/<engine>/seed_<k>
    parts = seed_dir.parts
    # Find last occurrence of "results" then pick relative pieces if possible
    # More robust: take last 4 components before seed
    rel = seed_dir
    # attempt: suite = parent of algorithm? from example: ZDT1/nsgaii/moocore/seed_7
    engine = seed_dir.parent.name
    algorithm = seed_dir.parent.parent.name if seed_dir.parent.parent else ""
    suite = seed_dir.parent.parent.parent.name if seed_dir.parent.parent.parent else ""
    return {"suite": suite, "algorithm": algorithm, "engine": engine}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="results/bench_smoke_engines", help="input campaign root")
    ap.add_argument("--output", type=str, default="artifacts/tidy/engine_smoke.csv", help="output tidy csv path")
    ap.add_argument("--campaign", type=str, default="bench_smoke_engines", help="campaign name stored in tidy")
    args = ap.parse_args()

    repo = Path.cwd()
    inp = (repo / args.input).resolve()
    outp = (repo / args.output).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        print("ERROR: input path not found:", inp)
        return 2

    seed_dirs = sorted([p for p in inp.rglob("seed_*") if p.is_dir() and (p / "metadata.json").exists()])
    if not seed_dirs:
        print("ERROR: no seed dirs found under:", inp)
        return 3

    rows: list[dict[str, Any]] = []
    all_keys: set[str] = set()

    for sd in seed_dirs:
        r: dict[str, Any] = {}
        r["run_path"] = sd.as_posix()
        r["campaign"] = args.campaign

        # infer suite/algorithm/engine from path
        inf = infer_from_path(sd)
        r.update(inf)

        meta = read_json(sd / "metadata.json")
        r["timestamp"] = safe_get(meta, "timestamp")
        r["git_revision"] = safe_get(meta, "git_revision")
        r["vamos_version"] = safe_get(meta, "vamos_version")

        # high-level fields if present
        for k in ["algorithm", "problem", "seed", "max_evaluations", "population_size"]:
            if k in meta:
                r[k] = meta[k]

        # backend/engine
        if "backend" in meta:
            r["backend"] = meta["backend"]
        if "backend_info" in meta and isinstance(meta["backend_info"], dict):
            tmp: dict[str, Any] = {}
            flatten("backend_info", meta["backend_info"], tmp)
            r.update(tmp)

        # problem details
        r["n_obj"] = find_first_key_recursively(meta, "n_obj")
        r["n_var"] = find_first_key_recursively(meta, "n_var")

        # config keys (shallow)
        if "config" in meta and isinstance(meta["config"], dict):
            # store some common config fields if available
            r["config.engine"] = find_first_key_recursively(meta["config"], "engine")
            r["config.selection"] = find_first_key_recursively(meta["config"], "selection")
            r["config.constraint_mode"] = find_first_key_recursively(meta["config"], "constraint_mode")

        # metrics passthrough (flatten)
        if "metrics" in meta and isinstance(meta["metrics"], dict):
            tmp: dict[str, Any] = {}
            flatten("metrics", meta["metrics"], tmp)
            r.update(tmp)

        # runtime from time.txt
        time_p = sd / "time.txt"
        if time_p.exists():
            t = try_parse_float(time_p.read_text(encoding="utf-8", errors="replace"))
            r["runtime_seconds"] = t
        else:
            r["runtime_seconds"] = None

        # FUN/X matrices
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
        else:
            r["front_size"] = None
            r["fun_ncols"] = None

        if x_p.exists():
            nrows_x, ncols_x, _ = read_csv_matrix(x_p)
            r["x_ncols"] = ncols_x
            r["x_nrows"] = nrows_x
        else:
            r["x_ncols"] = None
            r["x_nrows"] = None

        # Normalize engine field
        # prefer meta.backend, else config.engine, else path engine
        eng = r.get("backend") or r.get("config.engine") or r.get("engine")
        if eng is not None:
            r["engine"] = str(eng)

        rows.append(r)
        all_keys |= set(r.keys())

    # stable column order: core first, then the rest sorted
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
        "n_obj",
        "n_var",
        "runtime_seconds",
        "front_size",
        "fun_ncols",
        "x_nrows",
        "x_ncols",
        "git_revision",
        "timestamp",
        "vamos_version",
    ]
    rest = sorted([k for k in all_keys if k not in core])
    cols = core + rest

    with outp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("Wrote:", outp)
    print("Runs:", len(rows))
    print("Columns:", len(cols))
    print("Column sample:", cols[:25])
    # print first 6 lines
    lines = outp.read_text(encoding="utf-8").splitlines()
    print("\nPreview:")
    print("\n".join(lines[: min(6, len(lines))]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
