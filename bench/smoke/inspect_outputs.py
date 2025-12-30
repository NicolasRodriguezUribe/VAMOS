from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "bench_smoke_engines"

def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def head(path: Path, n: int = 25) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[:n])
    except Exception as e:
        return f"<cannot read: {e}>"

def summarize_meta(meta: dict) -> dict:
    # Intentamos extraer campos tipicos sin asumir estructura exacta
    keys = list(meta.keys())
    out = {"_top_keys": keys[:60]}
    # Campos comunes si existen
    for k in [
        "algorithm", "engine", "problem", "seed", "max_evaluations",
        "population_size", "offspring_population_size", "n_obj", "n_var",
        "start_time", "end_time", "elapsed_seconds", "runtime_seconds",
        "git_commit", "commit", "platform", "python", "versions"
    ]:
        if k in meta:
            out[k] = meta[k]
    # Si hay subestructuras tipicas, las mostramos someramente
    for k in ["environment", "timing", "timings", "run", "experiment", "config", "defaults"]:
        if k in meta and isinstance(meta[k], dict):
            out[f"{k}._keys"] = list(meta[k].keys())[:60]
    return out

def list_files(dirpath: Path) -> list[str]:
    files = []
    for p in sorted(dirpath.rglob("*")):
        if p.is_file():
            files.append(f"{p.relative_to(dirpath).as_posix()} ({p.stat().st_size} bytes)")
    return files

def main() -> int:
    if not OUT.exists():
        print("ERROR: output root not found:", OUT)
        return 2

    # Detect run folders: .../ZDT1/nsgaii/<engine>/seed_7/
    run_dirs = sorted([p for p in OUT.rglob("seed_*") if p.is_dir()])
    print("Found seed directories:", len(run_dirs))
    for rd in run_dirs:
        print("-", rd.relative_to(REPO).as_posix())

    print("\n=== PER-RUN INSPECTION (metadata + resolved_config + file list) ===")
    for rd in run_dirs:
        # infer engine from path (one level up)
        engine = rd.parent.name
        problem = rd.parents[2].name if len(rd.parents) >= 3 else "unknown"
        algo = rd.parents[1].name if len(rd.parents) >= 2 else "unknown"

        print("\n--- RUN ---")
        print("path:", rd.relative_to(REPO).as_posix())
        print("problem:", problem, "algo:", algo, "engine:", engine)

        meta_p = rd / "metadata.json"
        cfg_p = rd / "resolved_config.json"

        if meta_p.exists():
            meta = read_json(meta_p)
            sm = summarize_meta(meta)
            print("metadata.json summary:")
            print(json.dumps(sm, indent=2, ensure_ascii=False)[:6000])
        else:
            print("metadata.json: MISSING")

        if cfg_p.exists():
            cfg = read_json(cfg_p)
            # mostramos solo defaults + nsgaii si existen
            view = {}
            if isinstance(cfg, dict):
                if "defaults" in cfg:
                    view["defaults"] = cfg["defaults"]
                # algunos configs anidan por algoritmo
                for k in ["nsgaii", "moead", "smsemoa"]:
                    if k in cfg:
                        view[k] = cfg[k]
            print("resolved_config.json view (defaults + algo block if present):")
            print(json.dumps(view, indent=2, ensure_ascii=False)[:6000])
        else:
            print("resolved_config.json: MISSING")

        # file list
        files = list_files(rd)
        print("files under seed dir:", len(files))
        for f in files[:200]:
            print(" ", f)
        if len(files) > 200:
            print("  ... truncated")

        # quick peek at CSV heads if any
        csvs = [p for p in rd.rglob("*.csv") if p.is_file()]
        if csvs:
            print("\nCSV preview (first 25 lines each, up to 4 files):")
            for p in csvs[:4]:
                print("\n#", p.relative_to(REPO).as_posix())
                print(head(p, 25))
        else:
            print("\nNo CSV files found under this seed dir.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
