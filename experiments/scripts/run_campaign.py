from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

def load_yaml(path: Path) -> dict:
    import yaml  # type: ignore
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def dump_yaml(obj: dict, path: Path) -> None:
    import yaml  # type: ignore
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def deep_copy(x: Any) -> Any:
    return json.loads(json.dumps(x))

def flatten_seed_rule(rule: dict) -> List[int]:
    start = int(rule.get("start", 1))
    count = int(rule.get("count", 1))
    step = int(rule.get("step", 1))
    return [start + i * step for i in range(count)]

def problem_domain(problem_catalog: dict, problem_key: str) -> str:
    for p in problem_catalog.get("problems", []):
        if p.get("problem_key") == problem_key:
            return str(p.get("domain", "unknown"))
    return "unknown"

def keep_keys(d: dict, allowed: set[str]) -> dict:
    return {k: v for k, v in d.items() if k in allowed}

def set_aos_enabled(cfg: dict, problem: str, algo: str, enabled: bool) -> None:
    try:
        aos = cfg.setdefault("problems", {}).setdefault(problem, {}).setdefault(algo, {}).setdefault("adaptive_operator_selection", {})
        aos["enabled"] = bool(enabled)
    except Exception:
        pass

def base_template() -> dict:
    return {"defaults": {}, "problems": {}}

def compute_seed_dir(output_root: Path, suite: str, algo: str, engine: str, seed: int, problem_key: str) -> Path:
    # Mirror observed structure: results/<campaign>/<PROBLEM_KEY_UPPER>/<algo>/<engine>/seed_<k>/
    # The CLI uses the problem key (upper-cased) as the top-level directory.
    top = str(problem_key).upper()
    return output_root / top / algo / engine / f"seed_{seed}"

def infer_suite_from_problem(problem_key: str) -> str:
    # Use prefix heuristics similar to catalog
    import re
    k = str(problem_key).lower()
    m = re.match(r"^([a-z]+)", k)
    pref = m.group(1) if m else "unknown"
    # For display we keep uppercase for benchmark families (matches your smoke output style)
    return pref.upper()

@dataclass
class RunSpec:
    algo: str
    engine: str
    problem: str
    suite: str
    seed: int
    pop: int
    off: int
    maxeval: int
    output_root: Path
    cfg_path: Path
    log_path: Path
    seed_dir: Path

def build_config(
    *,
    algo: str,
    engine: str,
    problem: str,
    seed: int,
    pop: int,
    off: int,
    maxeval: int,
    output_root: Path,
    aos_enabled: bool,
    algo_keys: set[str],
    operator_block: dict,
    track_genealogy: Optional[bool],
) -> dict:
    cfg = base_template()
    cfg["defaults"] = {
        "algorithm": algo,
        "engine": engine,
        "problem": problem,
        "output_root": str(output_root.as_posix()),
        "population_size": pop,
        "offspring_population_size": off,
        "max_evaluations": maxeval,
        "seed": int(seed),
    }

    # Optional track_genealogy only if supported in metadata.config keys
    if track_genealogy is not None and "track_genealogy" in algo_keys:
        cfg["defaults"]["track_genealogy"] = bool(track_genealogy)

    # Apply operator_block keys ONLY if algo supports them (based on discovered keys)
    # Discovered keys live under meta["config"] (not defaults), but they correspond to algorithm config surface.
    # In your canonical configs, these are set under defaults.<algo> (which the CLI resolves into config).
    op_payload = {}
    for k in ["crossover", "mutation", "selection", "repair", "survival"]:
        if k in algo_keys and operator_block.get(k) is not None:
            op_payload[k] = deep_copy(operator_block[k])

    if op_payload:
        cfg["defaults"][algo] = op_payload

    # Always attempt to disable AOS for safety (will be ignored if unsupported)
    set_aos_enabled(cfg, problem, algo, aos_enabled)

    return cfg

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=str, required=True, help="Path to campaign spec YAML")
    ap.add_argument("--dry-run", action="store_true", help="Only generate configs and counts; do not run CLI")
    ap.add_argument("--resume", action="store_true", help="Skip runs that already have metadata.json in expected output dir")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of runs executed (0 = no limit)")
    ap.add_argument("--print-configs", type=int, default=3, help="Print first N generated config paths")
    args = ap.parse_args()

    repo = Path.cwd()
    spec_path = (repo / args.spec).resolve()
    spec = load_yaml(spec_path)

    campaign = spec["campaign"]
    output_root = (repo / spec["output_root"]).resolve()

    # Inputs
    inputs = spec.get("inputs", {})
    prob_catalog = read_json((repo / inputs["problem_catalog"]).resolve())
    algo_keys_map = read_json((repo / inputs["algo_config_keys"]).resolve())
    operator_blocks = load_yaml((repo / inputs["operator_blocks"]).resolve())

    common = spec.get("common", {})
    aos_enabled = bool(common.get("aos_enabled", False))
    track_genealogy = common.get("track_genealogy", None)

    matrix = spec["matrix"]
    engines = list(matrix["engines"])
    algos = list(matrix["algorithms"])
    seeds = flatten_seed_rule(matrix["seed_rule"])
    problems = list(matrix["problems"])

    budget = matrix["budget"]
    pop = int(budget["population_size"])
    off = int(budget["offspring_size"])
    maxeval = int(budget["max_evaluations"])

    dom_to_block_name = matrix.get("domain_operator_block", {})

    cfg_root = repo / "experiments" / "configs" / "generated" / campaign
    log_root = repo / "experiments" / "scripts" / "logs" / campaign
    ensure_dir(cfg_root)
    ensure_dir(log_root)
    ensure_dir(output_root)

    runs: List[RunSpec] = []
    for algo in algos:
        algo_keys = set(algo_keys_map.get(algo, []))
        for engine in engines:
            for problem in problems:
                suite = infer_suite_from_problem(problem)
                dom = problem_domain(prob_catalog, problem)
                block_name = dom_to_block_name.get(dom)
                if not block_name:
                    # Skip problems without operator block mapping (phase 2 handles discrete)
                    continue
                operator_block = operator_blocks.get(block_name, {})

                for seed in seeds:
                    seed_dir = compute_seed_dir(output_root, suite, algo, engine, seed, problem)
                    if args.resume and (seed_dir / "metadata.json").exists():
                        continue

                    cfg = build_config(
                        algo=algo, engine=engine, problem=problem, seed=seed,
                        pop=pop, off=off, maxeval=maxeval,
                        output_root=output_root,
                        aos_enabled=aos_enabled,
                        algo_keys=algo_keys,
                        operator_block=operator_block,
                        track_genealogy=track_genealogy,
                    )

                    cfg_name = f"{campaign}__{suite}__{algo}__{problem}__{engine}__seed{seed}.yml"
                    cfg_path = cfg_root / cfg_name
                    log_path = log_root / f"{cfg_path.stem}.log"
                    dump_yaml(cfg, cfg_path)

                    runs.append(RunSpec(
                        algo=algo, engine=engine, problem=problem, suite=suite, seed=seed,
                        pop=pop, off=off, maxeval=maxeval,
                        output_root=output_root,
                        cfg_path=cfg_path, log_path=log_path, seed_dir=seed_dir
                    ))

    # Breakdown
    by = {}
    for r in runs:
        by.setdefault(r.algo, {}).setdefault(r.engine, 0)
        by[r.algo][r.engine] += 1

    print("\n=== DRY SUMMARY ===")
    print("campaign:", campaign)
    print("spec:", spec_path.relative_to(repo))
    print("output_root:", output_root.relative_to(repo))
    print("runs_generated:", len(runs))
    print("resume:", bool(args.resume), "dry_run:", bool(args.dry_run), "limit:", args.limit)
    print("breakdown (algo -> engine -> n_runs):")
    print(json.dumps(by, indent=2, ensure_ascii=False))

    if args.print_configs:
        print("\nSample configs:")
        for r in runs[: args.print_configs]:
            print("-", r.cfg_path.relative_to(repo))

    if args.dry_run:
        return 0

    # Execute runs (with optional limit)
    executed = 0
    failed = 0

    # Index file for traceability
    index_path = output_root / "runs_index.jsonl"
    with index_path.open("a", encoding="utf-8") as idx:
        for r in runs:
            if args.limit and executed >= args.limit:
                break

            cmd = [sys.executable, "-m", "vamos.experiment.cli.main", "--config", str(r.cfg_path)]
            print(f"\n=== RUN {executed+1}/{min(len(runs), args.limit) if args.limit else len(runs)} ===")
            print("cmd:", " ".join(cmd))

            with r.log_path.open("w", encoding="utf-8") as f:
                p = subprocess.run(cmd, cwd=repo, stdout=f, stderr=subprocess.STDOUT)

            rec = {
                "algo": r.algo, "engine": r.engine, "problem": r.problem, "suite": r.suite, "seed": r.seed,
                "pop": r.pop, "off": r.off, "maxeval": r.maxeval,
                "config": str(r.cfg_path.relative_to(repo)),
                "log": str(r.log_path.relative_to(repo)),
                "seed_dir": str(r.seed_dir.relative_to(repo)),
                "returncode": p.returncode,
            }
            idx.write(json.dumps(rec, ensure_ascii=False) + "\n")
            idx.flush()

            executed += 1
            if p.returncode != 0:
                failed += 1
                print("FAILED:", rec)

    print("\n=== EXEC SUMMARY ===")
    print("executed:", executed, "failed:", failed)
    print("index:", index_path.relative_to(repo))
    return 0 if failed == 0 else 10

if __name__ == "__main__":
    raise SystemExit(main())
