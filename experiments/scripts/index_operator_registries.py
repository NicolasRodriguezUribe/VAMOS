from __future__ import annotations

import importlib
import json
import pkgutil
from pathlib import Path
from typing import Any

import vamos

REPO = Path.cwd()
OUT_JSON = REPO / "experiments" / "catalog" / "operator_registries.json"
OUT_SUM = REPO / "experiments" / "catalog" / "operator_registries_summary.json"

PATTERNS = ("operator", "operators", "variation", "crossover", "mutation", "repair", "selection")


def is_mapping_like(obj: Any) -> bool:
    return hasattr(obj, "keys") and hasattr(obj, "items")


def summarize_mapping(keys: list[Any], max_keys: int = 80) -> dict[str, Any]:
    sk = [k for k in keys if isinstance(k, (str, int, float))]
    sk2 = sorted([str(k) for k in sk])[:max_keys]
    return {"n_keys": len(keys), "sample_keys": sk2}


def main() -> int:
    registries: list[dict[str, Any]] = []

    for m in pkgutil.walk_packages(vamos.__path__, prefix=vamos.__name__ + "."):
        name = m.name
        if not any(p in name.lower() for p in PATTERNS):
            continue
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue

        for attr in dir(mod):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(mod, attr)
            except Exception:
                continue

            # mapping-like registries
            if is_mapping_like(val):
                try:
                    keys = list(val.keys())
                except Exception:
                    continue
                registries.append(
                    {
                        "module": name,
                        "attr": attr,
                        "summary": summarize_mapping(keys),
                    }
                )

            # objects that contain .REGISTRY or .registry dicts
            for subattr in ("REGISTRY", "registry"):
                if hasattr(val, subattr):
                    try:
                        sub = getattr(val, subattr)
                        if is_mapping_like(sub):
                            keys = list(sub.keys())
                            registries.append(
                                {
                                    "module": name,
                                    "attr": f"{attr}.{subattr}",
                                    "summary": summarize_mapping(keys),
                                }
                            )
                    except Exception:
                        pass

    # Deduplicate by (module, attr)
    seen = set()
    unique: list[dict[str, Any]] = []
    for r in registries:
        k = (r["module"], r["attr"])
        if k in seen:
            continue
        seen.add(k)
        unique.append(r)

    # Sort: largest registries first (use n_keys)
    unique.sort(key=lambda r: (-int(r["summary"]["n_keys"]), r["module"], r["attr"]))

    OUT_JSON.write_text(json.dumps(unique, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote:", OUT_JSON.relative_to(REPO), "registries:", len(unique))

    # Build a summary grouped by “kind” heuristics on module/attr name
    def kind(module: str, attr: str) -> str:
        t = f"{module}.{attr}".lower()
        if "crossover" in t:
            return "crossover"
        if "mutation" in t:
            return "mutation"
        if "repair" in t:
            return "repair"
        if "selection" in t:
            return "selection"
        if "survival" in t:
            return "survival"
        if "operator" in t or "variation" in t:
            return "operator/variation"
        return "other"

    by_kind: dict[str, list[dict[str, Any]]] = {}
    for r in unique:
        by_kind.setdefault(kind(r["module"], r["attr"]), []).append(r)

    summary = {
        "total_registries": len(unique),
        "kinds": {k: len(v) for k, v in sorted(by_kind.items(), key=lambda kv: (-len(kv[1]), kv[0]))},
        "top10_by_size": unique[:10],
    }

    OUT_SUM.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote:", OUT_SUM.relative_to(REPO))
    print("\nSummary:\n", json.dumps(summary, indent=2, ensure_ascii=False)[:12000])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
