from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from canonical_runs import component_name, result_runs


def dump_yaml(obj: object, path: Path) -> None:
    import yaml  # type: ignore[import-untyped]

    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")


def deep_copy(x: Any) -> Any:
    return json.loads(json.dumps(x))


def main() -> int:
    repo = Path.cwd()
    out_root = repo / "results" / "algo_schema_discovery"
    out_root.mkdir(parents=True, exist_ok=True)

    cfg_root = repo / "experiments" / "configs" / "generated" / "algo_schema_discovery"
    log_root = repo / "experiments" / "scripts" / "logs" / "algo_schema_discovery"
    cfg_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    algos = ["nsgaii", "nsgaiii", "moead", "smsemoa", "spea2", "ibea", "smpso"]
    engine = "numpy"
    problem = "zdt1"
    seed = 7

    # Use a known-good operator block for NSGA-II only (others we let defaults drive)
    nsgaii_block = {
        "crossover": {"method": "sbx", "prob": 0.9, "eta": 20},
        "mutation": {"method": "polynomial", "prob": "1/n", "eta": 20},
        "selection": {"method": "tournament", "size": 2},
    }

    runs: list[dict[str, Any]] = []
    for algo in algos:
        cfg: dict[str, Any] = {
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
            "problems": {problem: {algo: {"adaptive_operator_selection": {"enabled": False}}}},
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

        runs.append(
            {"algo": algo, "returncode": p.returncode, "config": str(cfg_path.relative_to(repo)), "log": str(log_path.relative_to(repo))}
        )

    print("\n=== RUN SUMMARY ===")
    for r in runs:
        print(r)

    canonical_runs = result_runs(out_root)
    print("\nFound canonical runs with results:", len(canonical_runs))
    by_algo: dict[str, dict[str, Any]] = {}
    for run in canonical_runs:
        resolved = run.manifest.resolved_spec
        algorithm = resolved.get("algorithm")
        backend = resolved.get("backend")
        algorithm_map = algorithm if isinstance(algorithm, Mapping) else {}
        backend_map = backend if isinstance(backend, Mapping) else {}
        kernel = backend_map.get("kernel")
        resolved_config = algorithm_map.get("config")
        config_map = cast(Mapping[str, Any], resolved_config) if isinstance(resolved_config, Mapping) else {}
        algo = component_name(algorithm_map)
        by_algo.setdefault(algo, {})
        by_algo[algo] = {
            "engine": component_name(kernel),
            "path": str(run.root.relative_to(repo)),
            "config_keys": sorted(str(key) for key in config_map)[:120],
            "manifest_keys": list(run.manifest)[:60],
        }

    print("\n=== CONFIG KEYS BY ALGORITHM (from canonical manifests) ===")
    print(json.dumps(by_algo, indent=2, ensure_ascii=False)[:12000])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
