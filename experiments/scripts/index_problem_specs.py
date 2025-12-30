from __future__ import annotations

import csv
import json
import re
import pkgutil
import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import vamos

REPO = Path.cwd()
OUT_DIR = REPO / "experiments" / "catalog"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV  = OUT_DIR / "problem_specs.csv"
OUT_SUM  = OUT_DIR / "catalog_summary.json"
OUT_JSON = OUT_DIR / "problem_specs_sanitized.json"

KEY_PATTERNS = ("problem", "registry", "spec")

def is_mapping_like(obj: Any) -> bool:
    return hasattr(obj, "keys") and hasattr(obj, "items")

def find_mapping_attr(attr_name: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    for m in pkgutil.walk_packages(vamos.__path__, prefix=vamos.__name__ + "."):
        name = m.name
        if not any(p in name.lower() for p in KEY_PATTERNS):
            continue
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(mod, attr_name):
            try:
                val = getattr(mod, attr_name)
                if is_mapping_like(val):
                    return name, val  # type: ignore[return-value]
            except Exception:
                continue
    return None, None

def infer_family(problem_key: str) -> str:
    k = str(problem_key).strip().lower()
    # take prefix until first digit or underscore
    m = re.match(r"^([a-z]+)", k)
    pref = m.group(1) if m else "unknown"
    # normalize common families
    known = {"zdt","dtlz","wfg","lz","cec","tsp","tsplib","real","real_world"}
    if pref in known:
        return "real_world" if pref == "real" else pref
    return pref or "unknown"

def find_first_int(obj: Any, candidates: tuple[str, ...]) -> Optional[int]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in candidates:
                try:
                    return int(v)
                except Exception:
                    pass
        for v in obj.values():
            got = find_first_int(v, candidates)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = find_first_int(v, candidates)
            if got is not None:
                return got
    return None

def sanitize(obj: Any) -> Any:
    # Make it JSON serializable without losing too much info
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    # fallback for callables / classes / numpy types
    return f"<{type(obj).__name__}>"

def main() -> int:
    modname, specs = find_mapping_attr("PROBLEM_SPECS")
    if specs is None:
        print("ERROR: Could not locate PROBLEM_SPECS mapping in vamos.* modules.")
        return 2

    # Try to extract n_obj/n_var with broad candidate key names
    n_obj = ("n_obj","n_objectives","num_objectives","objectives","m")
    n_var = ("n_var","n_variables","num_variables","variables","dimension","d","n_dim")

    rows = []
    fam_counts: Dict[str, int] = {}

    sanitized = {}
    for key, spec in specs.items():
        fam = infer_family(str(key))
        fam_counts[fam] = fam_counts.get(fam, 0) + 1

        spec_s = sanitize(spec)
        sanitized[str(key)] = spec_s

        m = find_first_int(spec_s, n_obj)
        d = find_first_int(spec_s, n_var)

        rows.append({
            "problem_key": key,
            "family": fam,
            "n_obj": "" if m is None else m,
            "n_var": "" if d is None else d,
        })

    rows.sort(key=lambda r: (str(r["family"]), str(r["problem_key"])))

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["problem_key","family","n_obj","n_var"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    OUT_JSON.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False), encoding="utf-8")

    # Provide sample keys per family
    samples: Dict[str, list[str]] = {}
    for r in rows:
        fam = str(r["family"])
        samples.setdefault(fam, [])
        if len(samples[fam]) < 12:
            samples[fam].append(str(r["problem_key"]))

    summary = {
        "problem_specs_module": modname,
        "problem_specs_count": len(rows),
        "families": dict(sorted(fam_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "sample_keys_by_family": samples,
        "csv_path": str(OUT_CSV.relative_to(REPO)),
        "sanitized_json_path": str(OUT_JSON.relative_to(REPO)),
    }
    OUT_SUM.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Wrote:", OUT_CSV.relative_to(REPO), "rows:", len(rows))
    print("Wrote:", OUT_JSON.relative_to(REPO))
    print("Wrote:", OUT_SUM.relative_to(REPO))
    print("\nSummary:\n", json.dumps(summary, indent=2, ensure_ascii=False))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
