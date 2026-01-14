from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("PyYAML is required for this script. Install with 'pip install pyyaml'.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def dump_yaml(obj: dict, path: Path) -> None:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("PyYAML is required for this script. Install with 'pip install pyyaml'.") from exc
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def deep_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def flatten_seed_rule(rule: dict) -> list[int]:
    start = int(rule.get("start", 1))
    count = int(rule.get("count", 1))
    step = int(rule.get("step", 1))
    return [start + i * step for i in range(count)]


def merge_dict(dst: dict, src: dict) -> dict:
    # recursive merge (src overrides)
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            merge_dict(dst[key], value)
        else:
            dst[key] = deep_copy(value)
    return dst


def infer_suite_from_problem(problem_key: str) -> str:
    import re

    key = str(problem_key).lower()
    match = re.match(r"^([a-z]+)", key)
    prefix = match.group(1) if match else "unknown"
    return prefix.upper()


def problem_domain(problem_catalog: dict, problem_key: str) -> str:
    for problem in problem_catalog.get("problems", []):
        if problem.get("problem_key") == problem_key:
            return str(problem.get("domain", "unknown"))
    return "unknown"


def build_config_base(
    *,
    algo: str,
    engine: str,
    problem: str,
    seed: int,
    pop: int,
    off: int,
    maxeval: int,
    output_root: Path,
    algo_keys: set[str],
    operator_block: dict,
    track_genealogy: bool | None,
) -> dict:
    cfg = {"defaults": {}, "problems": {}}
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
    if track_genealogy is not None and "track_genealogy" in algo_keys:
        cfg["defaults"]["track_genealogy"] = bool(track_genealogy)

    op_payload = {}
    for key in ["crossover", "mutation", "selection", "repair"]:
        if key in algo_keys and operator_block.get(key) is not None:
            op_payload[key] = deep_copy(operator_block[key])
    if op_payload:
        cfg["defaults"][algo] = op_payload

    # Disable AOS unless explicitly configured elsewhere
    cfg.setdefault("problems", {}).setdefault(problem, {}).setdefault(algo, {}).setdefault("adaptive_operator_selection", {})["enabled"] = (
        False
    )
    return cfg


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
    out_root: Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--print-configs", type=int, default=4)
    args = ap.parse_args()

    repo = Path.cwd()
    spec_path = (repo / args.spec).resolve()
    spec = load_yaml(spec_path)

    campaign = spec["campaign"]
    output_root_base = (repo / spec["output_root"]).resolve()

    inputs = spec.get("inputs", {})
    prob_catalog = read_json((repo / inputs["problem_catalog"]).resolve())
    algo_keys_map = read_json((repo / inputs["algo_config_keys"]).resolve())
    operator_blocks = load_yaml((repo / inputs["operator_blocks"]).resolve())

    common = spec.get("common", {})
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
    dom_to_block = matrix.get("domain_operator_block", {})

    variants = spec.get("variants", [])
    if not variants:
        raise SystemExit("Spec must define variants[]")

    cfg_root = repo / "experiments" / "configs" / "generated" / campaign
    log_root = repo / "experiments" / "scripts" / "logs" / campaign
    ensure_dir(cfg_root)
    ensure_dir(log_root)
    ensure_dir(output_root_base)

    runs: list[RunSpec] = []
    for variant in variants:
        vname = variant["name"]
        vpatch = variant.get("patch", {}) or {}
        v_out_root = output_root_base / vname
        ensure_dir(v_out_root)

        for algo in algos:
            algo_keys = set(algo_keys_map.get(algo, []))
            for engine in engines:
                for problem in problems:
                    suite = infer_suite_from_problem(problem)
                    domain = problem_domain(prob_catalog, problem)
                    block_name = dom_to_block.get(domain)
                    if not block_name:
                        continue
                    operator_block = operator_blocks.get(block_name, {})

                    for seed in seeds:
                        cfg = build_config_base(
                            algo=algo,
                            engine=engine,
                            problem=problem,
                            seed=seed,
                            pop=pop,
                            off=off,
                            maxeval=maxeval,
                            output_root=v_out_root,
                            algo_keys=algo_keys,
                            operator_block=operator_block,
                            track_genealogy=track_genealogy,
                        )
                        if isinstance(vpatch, dict) and vpatch:
                            merge_dict(cfg, vpatch)

                        cfg_name = f"{campaign}__{vname}__{suite}__{algo}__{problem}__{engine}__seed{seed}.yml"
                        cfg_path = cfg_root / cfg_name
                        log_path = log_root / f"{cfg_path.stem}.log"
                        dump_yaml(cfg, cfg_path)

                        if args.resume and log_path.exists():
                            continue

                        runs.append(
                            RunSpec(
                                variant=vname,
                                algo=algo,
                                engine=engine,
                                problem=problem,
                                suite=suite,
                                seed=seed,
                                cfg_path=cfg_path,
                                log_path=log_path,
                                out_root=v_out_root,
                            )
                        )

    summary = {}
    for run in runs:
        summary.setdefault(run.variant, {}).setdefault(run.algo, {}).setdefault(run.engine, 0)
        summary[run.variant][run.algo][run.engine] += 1

    print("\n=== VARIANT CAMPAIGN SUMMARY ===")
    print("campaign:", campaign)
    print("spec:", spec_path.relative_to(repo))
    print("output_root_base:", output_root_base.relative_to(repo))
    print("runs_generated:", len(runs))
    print("dry_run:", bool(args.dry_run), "resume:", bool(args.resume), "limit:", args.limit)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.print_configs:
        print("\nSample configs:")
        for run in runs[: args.print_configs]:
            print("-", run.cfg_path.relative_to(repo))

    if args.dry_run:
        return 0

    executed = 0
    failed = 0
    index_path = output_root_base / "runs_index.jsonl"
    with index_path.open("a", encoding="utf-8") as idx:
        for run in runs:
            if args.limit and executed >= args.limit:
                break
            cmd = [sys.executable, "-m", "vamos.experiment.cli.main", "--config", str(run.cfg_path)]
            with run.log_path.open("w", encoding="utf-8") as f:
                proc = subprocess.run(cmd, cwd=repo, stdout=f, stderr=subprocess.STDOUT)

            rec = {
                "campaign": campaign,
                "variant": run.variant,
                "algo": run.algo,
                "engine": run.engine,
                "problem": run.problem,
                "suite": run.suite,
                "seed": run.seed,
                "config": str(run.cfg_path.relative_to(repo)),
                "log": str(run.log_path.relative_to(repo)),
                "returncode": proc.returncode,
            }
            idx.write(json.dumps(rec, ensure_ascii=False) + "\n")
            idx.flush()

            executed += 1
            if proc.returncode != 0:
                failed += 1

    print("\n=== EXEC SUMMARY ===")
    print("executed:", executed, "failed:", failed)
    print("index:", index_path.relative_to(repo))
    return 0 if failed == 0 else 10


if __name__ == "__main__":
    raise SystemExit(main())
