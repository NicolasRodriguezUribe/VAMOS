from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

def load_yaml(path: Path) -> dict:
    import yaml  # type: ignore
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def dump_yaml(obj: dict, path: Path) -> None:
    import yaml  # type: ignore
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")

def deep_copy(x: Any) -> Any:
    return json.loads(json.dumps(x))

def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def find_seed_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("seed_*") if p.is_dir() and (p / "metadata.json").exists()])

def infer_algo_engine(sd: Path) -> Tuple[str,str]:
    eng = sd.parent.name
    algo = sd.parent.parent.name if sd.parent.parent else "unknown"
    return algo, eng

def main() -> int:
    repo = Path.cwd()
    out_root = repo / "results" / "algo_schema_discovery"
    out_root.mkdir(parents=True, exist_ok=True)

    cfg_root = repo / "experiments" / "configs" / "generated" / "algo_schema_discovery"
    log_root = repo / "experiments" / "scripts" / "logs" / "algo_schema_discovery"
    cfg_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    algos = ["nsgaii","nsgaiii","moead","smsemoa","spea2","ibea","smpso"]
    engine = "numpy"
    problem = "zdt1"
    seed = 7

    # Use a known-good operator block for NSGA-II only (others we let defaults drive)
    nsgaii_block = {
        "crossover": {"method": "sbx", "prob": 0.9, "eta": 20},
        "mutation":  {"method": "pm",  "prob": "1/n", "eta": 20},
        "selection": {"method": "tournament", "pressure": 2},
    }

    runs = []
    for algo in algos:
        cfg = {
            "defaults": {
                "algorithm": algo,
                "engine": engine,
                "problem": problem,
                "output_root": str(out_root.as_posix()),
                "population_size": 64,
                "offspring_population_size": 64,
                "max_evaluations": 800,
                "seed": seed,
                "selection_pressure": 2,
            },
            "problems": {
                problem: {
                    algo: {
                        "adaptive_operator_selection": {"enabled": False}
                    }
                }
            }
        }
        if algo == "nsgaii":
            cfg["defaults"]["nsgaii"] = deep_copy(nsgaii_block)

        cfg_path = cfg_root / f"{algo}__{problem}__{engine}__seed{seed}.yml"
        dump_yaml(cfg, cfg_path)

        log_path = log_root / f"{cfg_path.stem}.log"
        cmd = [sys.executable, "-m", "vamos.experiment.cli.main", "--config", str(cfg_path)]
        print("\n=== RUN:", algo, "===")
        print("cmd:", " ".join(cmd))
        with log_path.open("w", encoding="utf-8") as f:
            p = subprocess.run(cmd, cwd=repo, stdout=f, stderr=subprocess.STDOUT)

        runs.append({"algo": algo, "returncode": p.returncode, "config": str(cfg_path.relative_to(repo)), "log": str(log_path.relative_to(repo))})

    print("\n=== RUN SUMMARY ===")
    for r in runs:
        print(r)

    # Inspect metadata.config._keys for successes
    seed_dirs = find_seed_dirs(out_root)
    print("\nFound seed dirs:", len(seed_dirs))
    by_algo: Dict[str, dict] = {}
    for sd in seed_dirs:
        algo, eng = infer_algo_engine(sd)
        meta = read_json(sd / "metadata.json")
        cfg = meta.get("config", {})
        keys = list(cfg.keys()) if isinstance(cfg, dict) else []
        by_algo.setdefault(algo, {})
        by_algo[algo] = {
            "engine": eng,
            "path": str(sd.relative_to(repo)),
            "config_keys": keys[:120],
            "top_meta_keys": list(meta.keys())[:60],
        }

    print("\n=== CONFIG KEYS BY ALGORITHM (from metadata.json) ===")
    print(json.dumps(by_algo, indent=2, ensure_ascii=False)[:12000])

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
