from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

def load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise SystemExit("Missing dependency: PyYAML. Install with: pip install pyyaml") from e
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def dump_yaml(obj: dict, path: Path) -> None:
    import yaml  # type: ignore
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")

def tree(root: Path, max_files: int = 400) -> list[str]:
    out = []
    n = 0
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(root)
        size = p.stat().st_size
        out.append(f"{rel.as_posix()}  ({size} bytes)")
        n += 1
        if n >= max_files:
            out.append(f"... truncated at {max_files} files")
            break
    return out

def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    base_cfg = repo / "examples" / "configs" / "nsgaii_aos_min.yml"
    if not base_cfg.exists():
        print("ERROR: base config not found:", base_cfg)
        return 2

    out_root = repo / "results" / "bench_smoke_engines"
    cfg_root = repo / "bench" / "smoke" / "configs"
    log_root = repo / "bench" / "smoke" / "logs"
    for d in (cfg_root, log_root, out_root):
        d.mkdir(parents=True, exist_ok=True)

    base = load_yaml(base_cfg)

    # Robustly disable AOS if present (we want pure engine effect here)
    try:
        base["problems"]["zdt1"]["nsgaii"]["adaptive_operator_selection"]["enabled"] = False
    except Exception:
        pass

    # Make the run non-trivial but still “smoke”
    base["defaults"]["problem"] = "zdt1"
    base["defaults"]["population_size"] = 64
    base["defaults"]["offspring_population_size"] = 64
    base["defaults"]["max_evaluations"] = 2000
    base["defaults"]["seed"] = 7
    base["defaults"]["output_root"] = str(out_root.as_posix())

    engines = ["numpy", "numba", "moocore"]

    results = []
    for eng in engines:
        cfg = json.loads(json.dumps(base))  # deep copy without extra deps
        cfg["defaults"]["engine"] = eng
        cfg_name = f"nsgaii_zdt1_{eng}.yml"
        cfg_path = cfg_root / cfg_name
        dump_yaml(cfg, cfg_path)

        log_path = log_root / f"run_{eng}.log"
        cmd = [sys.executable, "-m", "vamos.experiment.cli.main", "--config", str(cfg_path)]
        print("\n=== RUN:", eng, "===")
        print("cmd:", " ".join(cmd))

        with log_path.open("w", encoding="utf-8") as f:
            p = subprocess.run(cmd, cwd=repo, stdout=f, stderr=subprocess.STDOUT)

        results.append({"engine": eng, "returncode": p.returncode, "log": str(log_path)})

    print("\n=== SUMMARY ===")
    for r in results:
        print(r)

    print("\n=== OUTPUT TREE ===")
    if out_root.exists():
        for line in tree(out_root):
            print(line)
    else:
        print("No output root created:", out_root)

    # Heuristic: print the first metadata-like file if present
    meta_candidates = []
    if out_root.exists():
        for p in out_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".json", ".yml", ".yaml", ".csv"):
                name = p.name.lower()
                if "meta" in name or "config" in name or "summary" in name:
                    meta_candidates.append(p)
    meta_candidates = sorted(meta_candidates)[:8]

    if meta_candidates:
        print("\n=== METADATA-LIKE FILES (sample) ===")
        for p in meta_candidates:
            rel = p.relative_to(repo)
            print("-", rel.as_posix(), f"({p.stat().st_size} bytes)")
    else:
        print("\nNo obvious metadata-like files found yet (json/yml/csv with meta/config/summary in name).")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
