from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


def load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise SystemExit("Missing dependency: PyYAML. Install with: pip install pyyaml") from e
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_yaml(obj: dict, path: Path) -> None:
    import yaml  # type: ignore

    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")


def deep_copy(x: Any) -> Any:
    return json.loads(json.dumps(x))


def set_aos_enabled(cfg: dict, problem: str, algo: str, enabled: bool) -> None:
    # Try to toggle AOS in the per-problem override block if present
    try:
        aos = cfg.setdefault("problems", {}).setdefault(problem, {}).setdefault(algo, {}).setdefault("adaptive_operator_selection", {})
        aos["enabled"] = bool(enabled)
    except Exception:
        pass


def base_template() -> dict:
    # Minimal template compatible with vamos.experiment.cli.main configs
    return {
        "defaults": {
            "algorithm": None,
            "engine": None,
            "problem": None,
            "output_root": None,
            "population_size": None,
            "offspring_population_size": None,
            "max_evaluations": None,
            "seed": None,
            "selection_pressure": 2,
        },
        "problems": {},
    }


def main() -> int:
    repo = Path.cwd()
    spec_path = repo / "experiments" / "configs" / "engine_study_pilot.yml"
    spec = load_yaml(spec_path)

    campaign = spec["campaign"]
    out_root = repo / spec["output_root"]
    out_root.mkdir(parents=True, exist_ok=True)

    cfg_root = repo / "experiments" / "configs" / "generated" / campaign
    log_root = repo / "experiments" / "scripts" / "logs" / campaign
    cfg_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    engines: List[str] = spec["matrix"]["engines"]
    algos: List[str] = spec["matrix"]["algorithms"]
    seeds: List[int] = spec["matrix"]["seeds"]
    problems: List[dict] = spec["matrix"]["problems"]

    aos_enabled = bool(spec.get("common", {}).get("aos_enabled", False))
    selection_pressure = int(spec.get("common", {}).get("selection_pressure", 2))

    # Algo blocks (operators)
    algo_blocks = {k: v for k, v in spec["matrix"].items() if k not in ("engines", "algorithms", "seeds", "problems")}

    runs = []
    total = 0

    for algo in algos:
        for eng in engines:
            for pr in problems:
                prob = pr["problem"]
                pop = int(pr["population_size"])
                off = int(pr["offspring_size"])
                maxeval = int(pr["max_evaluations"])
                for seed in seeds:
                    total += 1
                    cfg = base_template()
                    cfg["defaults"].update(
                        {
                            "algorithm": algo,
                            "engine": eng,
                            "problem": prob,
                            "output_root": str(out_root.as_posix()),
                            "population_size": pop,
                            "offspring_population_size": off,
                            "max_evaluations": maxeval,
                            "seed": int(seed),
                            "selection_pressure": selection_pressure,
                        }
                    )

                    # Add algo operator block at top-level defaults.<algo> if your config schema expects it there.
                    # In your example it is: defaults: { nsgaii: { ... } }
                    if algo in algo_blocks:
                        cfg["defaults"][algo] = deep_copy(algo_blocks[algo])

                    # Ensure AOS disabled unless explicitly enabled
                    set_aos_enabled(cfg, prob, algo, aos_enabled)

                    cfg_name = f"{campaign}__{algo}__{prob}__{eng}__seed{seed}.yml"
                    cfg_path = cfg_root / cfg_name
                    dump_yaml(cfg, cfg_path)

                    log_path = log_root / f"{cfg_name}.log"
                    cmd = [sys.executable, "-m", "vamos.experiment.cli.main", "--config", str(cfg_path)]
                    print(f"\n=== RUN {total} / (unknown total at print-time) ===")
                    print("cmd:", " ".join(cmd))

                    with log_path.open("w", encoding="utf-8") as f:
                        p = subprocess.run(cmd, cwd=repo, stdout=f, stderr=subprocess.STDOUT)

                    runs.append(
                        {
                            "algo": algo,
                            "engine": eng,
                            "problem": prob,
                            "seed": seed,
                            "pop": pop,
                            "off": off,
                            "maxeval": maxeval,
                            "returncode": p.returncode,
                            "config": str(cfg_path.relative_to(repo)),
                            "log": str(log_path.relative_to(repo)),
                        }
                    )

    # Summary
    failed = [r for r in runs if r["returncode"] != 0]
    print("\n=== SUMMARY ===")
    print("runs:", len(runs))
    print("failed:", len(failed))
    if failed:
        print("First failures:")
        for r in failed[:6]:
            print(r)

    # Write an index JSON for traceability
    index_path = out_root / "runs_index.json"
    index_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print("Wrote:", index_path.relative_to(repo))

    return 0 if not failed else 10


if __name__ == "__main__":
    raise SystemExit(main())
