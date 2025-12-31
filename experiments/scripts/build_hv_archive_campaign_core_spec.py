from __future__ import annotations

import json
from pathlib import Path


REPO = Path.cwd()


def main() -> int:
    cat_path = REPO / "experiments" / "catalog" / "problem_catalog_enriched.json"
    if not cat_path.exists():
        raise SystemExit(f"Missing {cat_path}. Run your catalog builder first.")
    catalog = json.loads(cat_path.read_text(encoding="utf-8"))

    probs = [p for p in catalog.get("problems", []) if p.get("domain") == "real"]

    def by_family(family: str) -> list[str]:
        return [p["problem_key"] for p in probs if p.get("family") == family]

    zdt = by_family("zdt")[:3]
    dtlz = by_family("dtlz")[:2]
    wfg = by_family("wfg")[:2]
    lz = by_family("lz")[:1]
    cec = by_family("cec")[:1]

    chosen: list[str] = []
    for group in (zdt, dtlz, wfg, lz, cec):
        for key in group:
            if key not in chosen:
                chosen.append(key)

    if len(chosen) < 6:
        raise SystemExit(f"Too few problems selected ({len(chosen)}). Check catalog contents.")

    out = REPO / "experiments" / "configs" / "hv_archive_campaign_core.yml"
    out.parent.mkdir(parents=True, exist_ok=True)

    spec = {
        "campaign": "hv_archive_campaign_core",
        "output_root": "results/hv_archive_campaign_core",
        "inputs": {
            "problem_catalog": "experiments/catalog/problem_catalog_enriched.json",
            "algo_config_keys": "experiments/catalog/algo_config_keys.json",
            "operator_blocks": "experiments/catalog/operator_blocks.yml",
        },
        "common": {
            "track_genealogy": False,
        },
        "matrix": {
            "engines": ["numpy", "numba", "moocore"],
            "algorithms": ["nsgaii", "moead", "smsemoa", "spea2", "ibea", "nsgaiii", "smpso"],
            "seed_rule": {"start": 1, "count": 10, "step": 1},
            "budget": {"population_size": 200, "offspring_size": 200, "max_evaluations": 40000},
            "problems": chosen,
            "domain_operator_block": {"real": "real"},
        },
        "variants": [
            {
                "name": "baseline",
                "patch": {
                    "stopping": {"hv_convergence": {"enabled": False}},
                    "archive": {"bounded": {"enabled": False}},
                },
            },
            {
                "name": "archive_sizecap",
                "patch": {
                    "stopping": {"hv_convergence": {"enabled": False}},
                    "archive": {
                        "bounded": {
                            "enabled": True,
                            "archive_type": "size_cap",
                            "size_cap": 400,
                            "nondominated_only": True,
                            "prune_policy": "crowding",
                        }
                    },
                },
            },
            {
                "name": "archive_epsgrid",
                "patch": {
                    "stopping": {"hv_convergence": {"enabled": False}},
                    "archive": {
                        "bounded": {
                            "enabled": True,
                            "archive_type": "epsilon_grid",
                            "size_cap": 400,
                            "epsilon": 0.01,
                            "nondominated_only": True,
                        }
                    },
                },
            },
            {
                "name": "stop_only",
                "patch": {
                    "stopping": {
                        "hv_convergence": {
                            "enabled": True,
                            "every_k": 200,
                            "window": 10,
                            "patience": 5,
                            "epsilon": 1e-4,
                            "epsilon_mode": "rel",
                            "statistic": "median",
                            "min_points": 25,
                            "confidence": None,
                            "ref_point": "auto",
                        }
                    },
                    "archive": {"bounded": {"enabled": False}},
                },
            },
            {
                "name": "stop_plus_archive",
                "patch": {
                    "stopping": {
                        "hv_convergence": {
                            "enabled": True,
                            "every_k": 200,
                            "window": 10,
                            "patience": 5,
                            "epsilon": 1e-4,
                            "epsilon_mode": "rel",
                            "statistic": "median",
                            "min_points": 25,
                            "confidence": None,
                            "ref_point": "auto",
                        }
                    },
                    "archive": {
                        "bounded": {
                            "enabled": True,
                            "archive_type": "size_cap",
                            "size_cap": 400,
                            "nondominated_only": True,
                            "prune_policy": "crowding",
                        }
                    },
                },
            },
        ],
    }

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("PyYAML is required for this script. Install with 'pip install pyyaml'.") from exc
    out.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    print("Wrote:", out.relative_to(REPO))
    print("Problems chosen:", chosen)
    print("Variants:", [v["name"] for v in spec["variants"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
