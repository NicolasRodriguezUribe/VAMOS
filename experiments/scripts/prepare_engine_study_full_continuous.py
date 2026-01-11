from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path.cwd()

PROB_CSV = REPO / "experiments" / "catalog" / "problem_specs.csv"
PROB_JSON = REPO / "experiments" / "catalog" / "problem_specs_sanitized.json"
OPREG_JSON = REPO / "experiments" / "catalog" / "operator_registries.json"

OUT_PROB_ENRICH = REPO / "experiments" / "catalog" / "problem_catalog_enriched.json"
OUT_OP_BLOCKS = REPO / "experiments" / "catalog" / "operator_blocks.yml"
OUT_ALGO_KEYS = REPO / "experiments" / "catalog" / "algo_config_keys.json"
OUT_SPEC = REPO / "experiments" / "configs" / "engine_study_full_continuous.yml"


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def load_yaml_deps():
    try:
        import yaml  # type: ignore

        return yaml
    except Exception as e:
        raise SystemExit("Missing dependency: PyYAML. Install with: pip install pyyaml") from e


def dump_yaml(obj: dict, path: Path) -> None:
    yaml = load_yaml_deps()
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")


def infer_family(problem_key: str) -> str:
    k = str(problem_key).strip().lower()
    m = re.match(r"^([a-z]+)", k)
    pref = m.group(1) if m else "unknown"
    known = {"zdt", "dtlz", "wfg", "lz", "cec", "tsp", "tsplib", "bin", "int", "mixed", "real_world", "ml", "welded", "fs"}
    if pref in known:
        return pref
    # Treat common QAP/flowshop instance prefixes as permutation-like families later (phase 2).
    return pref or "unknown"


def infer_domain(problem_key: str, family: str) -> str:
    k = str(problem_key).lower()
    fam = family.lower()
    if fam in ("zdt", "dtlz", "wfg", "lz", "cec", "real_world", "welded", "ml"):
        return "real"
    if fam in ("bin",):
        return "bin"
    if fam in ("int",):
        return "int"
    if fam in ("mixed",):
        return "mixed"
    if fam in ("tsp", "tsplib", "fs"):
        return "perm"
    # Heuristic: QAPLIB-style instance names
    if k in ("kroa", "krob", "kroc", "krod", "kroe"):
        return "perm"
    return "unknown"


def extract_algo_config_keys(results_root: Path) -> Dict[str, List[str]]:
    # Parse metadata.json from results/algo_schema_discovery/**/seed_*/metadata.json
    out: Dict[str, List[str]] = {}
    if not results_root.exists():
        return out
    for sd in sorted([p for p in results_root.rglob("seed_*") if p.is_dir() and (p / "metadata.json").exists()]):
        meta = read_json(sd / "metadata.json")
        algo = str(meta.get("algorithm") or sd.parent.parent.name)
        cfg = meta.get("config", {})
        keys = list(cfg.keys()) if isinstance(cfg, dict) else []
        out[algo] = sorted(set(keys))
    return out


def choose_real_operator_block() -> dict:
    # We choose a conservative, known-good block (from your canonical examples)
    return {
        "crossover": {"method": "sbx", "prob": 0.9, "eta": 20},
        "mutation": {"method": "pm", "prob": "1/n", "eta": 20},
        "selection": {"method": "tournament", "pressure": 2},
        "repair": None,
    }


def main() -> int:
    if not PROB_CSV.exists() or not PROB_JSON.exists():
        print("ERROR: Missing problem catalog inputs. Expected:")
        print("-", PROB_CSV)
        print("-", PROB_JSON)
        return 2

    # Load problem keys from sanitized JSON (authoritative list)
    prob_specs = read_json(PROB_JSON)
    problem_keys = sorted(list(prob_specs.keys()))

    # Build enriched problem catalog
    problems: List[dict] = []
    fam_counts: Dict[str, int] = {}
    dom_counts: Dict[str, int] = {}
    for pk in problem_keys:
        fam = infer_family(pk)
        dom = infer_domain(pk, fam)
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
        dom_counts[dom] = dom_counts.get(dom, 0) + 1
        problems.append({"problem_key": pk, "family": fam, "domain": dom})

    OUT_PROB_ENRICH.write_text(
        json.dumps(
            {
                "counts_by_family": dict(sorted(fam_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                "counts_by_domain": dict(sorted(dom_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                "problems": problems,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("Wrote:", OUT_PROB_ENRICH.relative_to(REPO))
    print("Family counts:", dict(sorted(fam_counts.items(), key=lambda kv: (-kv[1], kv[0]))))
    print("Domain counts:", dict(sorted(dom_counts.items(), key=lambda kv: (-kv[1], kv[0]))))

    # Extract algo config keys from discovery runs (so we only set keys that exist)
    algo_keys = extract_algo_config_keys(REPO / "results" / "algo_schema_discovery")
    OUT_ALGO_KEYS.write_text(json.dumps(algo_keys, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote:", OUT_ALGO_KEYS.relative_to(REPO), "algorithms:", len(algo_keys))

    # Operator blocks (phase 1 = real only; discrete handled in phase 2)
    blocks = {
        "real": choose_real_operator_block(),
        # placeholders: to be validated via operator probing in phase 2
        "bin": {"crossover": None, "mutation": None, "selection": {"method": "tournament", "pressure": 2}, "repair": None},
        "int": {"crossover": None, "mutation": None, "selection": {"method": "tournament", "pressure": 2}, "repair": None},
        "perm": {"crossover": None, "mutation": None, "selection": {"method": "tournament", "pressure": 2}, "repair": None},
        "mixed": {"crossover": None, "mutation": None, "selection": {"method": "tournament", "pressure": 2}, "repair": None},
    }
    dump_yaml(blocks, OUT_OP_BLOCKS)
    print("Wrote:", OUT_OP_BLOCKS.relative_to(REPO))

    # Build FULL CONTINUOUS spec (maximal within continuous families)
    continuous_families = ["zdt", "dtlz", "wfg", "lz", "cec", "real_world", "welded", "ml"]
    cont_problems = [p["problem_key"] for p in problems if p["family"] in continuous_families and p["domain"] == "real"]

    spec = {
        "campaign": "engine_study_full_continuous",
        "output_root": "results/engine_study_full_continuous",
        "inputs": {
            "problem_catalog": str(OUT_PROB_ENRICH.relative_to(REPO)),
            "algo_config_keys": str(OUT_ALGO_KEYS.relative_to(REPO)),
            "operator_blocks": str(OUT_OP_BLOCKS.relative_to(REPO)),
        },
        "common": {
            "aos_enabled": False,
            "track_genealogy": False,
        },
        "matrix": {
            "engines": ["numpy", "numba", "moocore"],
            "algorithms": ["nsgaii", "nsgaiii", "moead", "smsemoa", "spea2", "ibea", "smpso"],
            # seed rule expands to a list at runtime (runner will expand it)
            "seed_rule": {"start": 1, "count": 30, "step": 1},
            # include all continuous problems by key (no guessing)
            "problems": cont_problems,
            # budget regimes (runner can select based on n_obj/n_var later; for now fixed per run)
            "budget": {"population_size": 200, "offspring_size": 200, "max_evaluations": 40000},
            # operator block selection by domain
            "domain_operator_block": {"real": "real"},
        },
    }
    dump_yaml(spec, OUT_SPEC)
    print("Wrote:", OUT_SPEC.relative_to(REPO))
    print("Continuous problems:", len(cont_problems))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
