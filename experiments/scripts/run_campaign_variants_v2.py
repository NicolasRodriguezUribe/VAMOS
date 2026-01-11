from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

def merge_dict(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            merge_dict(dst[k], v)
        else:
            dst[k] = deep_copy(v)
    return dst

def flatten_seed_rule(rule: dict) -> List[int]:
    start = int(rule.get("start", 1))
    count = int(rule.get("count", 1))
    step = int(rule.get("step", 1))
    return [start + i * step for i in range(count)]

def infer_suite(problem_key: str) -> str:
    import re
    k = str(problem_key).lower()
    m = re.match(r"^([a-z]+)", k)
    return (m.group(1) if m else "unknown").upper()

def load_success_set(index_path: Path) -> Set[str]:
    ok: Set[str] = set()
    if not index_path.exists():
        return ok
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if int(r.get("returncode", 1)) == 0:
            # unique key = config path (stable)
            ok.add(str(r.get("config", "")))
    return ok

@dataclass
class RunSpec:
    variant: str
    algo: str
    engine: str
    problem: str
    suite: str
    seed: int
    cfg_path: Path
    log_path: Path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--algo", action="append", default=[])
    ap.add_argument("--engine", action="append", default=[])
    ap.add_argument("--problem", action="append", default=[])
    ap.add_argument("--seed", type=int, action="append", default=[])
    args = ap.parse_args()

    repo = Path.cwd()
    spec_path = (repo / args.spec).resolve()
    spec = load_yaml(spec_path)

    campaign = spec["campaign"]
    output_root_base = (repo / spec["output_root"]).resolve()
    ensure_dir(output_root_base)
    index_path = output_root_base / "runs_index.jsonl"

    inputs = spec.get("inputs", {})
    prob_catalog = read_json((repo / inputs["problem_catalog"]).resolve())
    algo_keys_map = read_json((repo / inputs["algo_config_keys"]).resolve())
    operator_blocks = load_yaml((repo / inputs["operator_blocks"]).resolve())

    # domain lookup (real/bin/etc.)
    dom_map = {p["problem_key"]: p.get("domain","unknown") for p in prob_catalog.get("problems", [])}

    matrix = spec["matrix"]
    engines = list(matrix["engines"])
    algos = list(matrix["algorithms"])
    seeds = flatten_seed_rule(matrix["seed_rule"])
    problems = list(matrix["problems"])
    budget = matrix["budget"]
    pop = int(budget["population_size"])
    off = int(budget["offspring_size"])
    maxeval = int(budget["max_evaluations"])
    dom_to_block = matrix.get("domain_operator_block", {})

    variants = spec.get("variants", [])
    if not variants:
        raise SystemExit("Spec must define variants[]")

    # filters
    f_variants = set(args.variant)
    f_algos = set(args.algo)
    f_engines = set(args.engine)
    f_problems = set(args.problem)
    f_seeds = set(args.seed)

    ok_configs = load_success_set(index_path) if args.resume else set()

    cfg_root = repo / "experiments" / "configs" / "generated" / campaign
    log_root = repo / "experiments" / "scripts" / "logs" / campaign
    ensure_dir(cfg_root); ensure_dir(log_root)

    runs: List[RunSpec] = []
    for v in variants:
        vname = v["name"]
        if f_variants and vname not in f_variants:
            continue
        vpatch = v.get("patch", {}) or {}

        v_out_root = output_root_base / vname
        ensure_dir(v_out_root)

        for algo in algos:
            if f_algos and algo not in f_algos:
                continue
            algo_keys = set(algo_keys_map.get(algo, []))

            for engine in engines:
                if f_engines and engine not in f_engines:
                    continue

                for problem in problems:
                    if f_problems and problem not in f_problems:
                        continue

                    dom = dom_map.get(problem, "unknown")
                    block_name = dom_to_block.get(dom)
                    if not block_name:
                        continue
                    operator_block = operator_blocks.get(block_name, {})

                    for seed in seeds:
                        if f_seeds and seed not in f_seeds:
                            continue

                        cfg = {"defaults": {}, "problems": {}}
                        cfg["defaults"] = {
                            "algorithm": algo,
                            "engine": engine,
                            "problem": problem,
                            "output_root": str(v_out_root.as_posix()),
                            "population_size": pop,
                            "offspring_population_size": off,
                            "max_evaluations": maxeval,
                            "seed": int(seed),
                        }

                        # operator block only if supported
                        op_payload = {}
                        for k in ["crossover", "mutation", "selection", "repair"]:
                            if k in algo_keys and operator_block.get(k) is not None:
                                op_payload[k] = deep_copy(operator_block[k])
                        if op_payload:
                            cfg["defaults"][algo] = op_payload

                        # disable AOS explicitly
                        cfg.setdefault("problems", {}).setdefault(problem, {}).setdefault(algo, {}).setdefault("adaptive_operator_selection", {})["enabled"] = False

                        # apply variant patch (stopping/archive)
                        if isinstance(vpatch, dict) and vpatch:
                            merge_dict(cfg, vpatch)

                        suite = infer_suite(problem)
                        cfg_name = f"{campaign}__{vname}__{suite}__{algo}__{problem}__{engine}__seed{seed}.yml"
                        cfg_path = cfg_root / cfg_name
                        log_path = log_root / f"{cfg_path.stem}.log"
                        dump_yaml(cfg, cfg_path)

                        cfg_rel = str(cfg_path.relative_to(repo))
                        if args.resume and cfg_rel in ok_configs:
                            continue

                        runs.append(RunSpec(vname, algo, engine, problem, suite, seed, cfg_path, log_path))

    print("campaign:", campaign)
    print("spec:", spec_path.relative_to(repo))
    print("runs_generated:", len(runs))
    print("resume:", bool(args.resume), "limit:", args.limit)

    if args.dry_run:
        return 0

    executed = 0
    failed = 0
    with index_path.open("a", encoding="utf-8") as idx:
        for r in runs:
            if args.limit and executed >= args.limit:
                break
            cmd = [sys.executable, "-m", "vamos.experiment.cli.main", "--config", str(r.cfg_path)]
            with r.log_path.open("w", encoding="utf-8") as f:
                p = subprocess.run(cmd, cwd=repo, stdout=f, stderr=subprocess.STDOUT)

            rec = {
                "campaign": campaign,
                "variant": r.variant,
                "algo": r.algo,
                "engine": r.engine,
                "problem": r.problem,
                "suite": r.suite,
                "seed": r.seed,
                "config": str(r.cfg_path.relative_to(repo)),
                "log": str(r.log_path.relative_to(repo)),
                "returncode": p.returncode,
            }
            idx.write(json.dumps(rec, ensure_ascii=False) + "\n")
            idx.flush()

            executed += 1
            if p.returncode != 0:
                failed += 1

    print("executed:", executed, "failed:", failed)
    print("index:", index_path.relative_to(repo))
    return 0 if failed == 0 else 10

if __name__ == "__main__":
    raise SystemExit(main())
