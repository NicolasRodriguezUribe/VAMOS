"""
Compare baseline vs after summaries for the native C++ refactor benchmark.

Usage:
  python paper/31_compare_cpp_native_refactor_results.py

Environment variables:
  - VAMOS_CPP_COMPARE_BASELINE: path to baseline_summary.json
      (default: results/cpp_native_refactor_baseline/baseline_summary.json)
  - VAMOS_CPP_COMPARE_AFTER: path to after_summary.json
      (default: results/cpp_native_refactor_after/after_summary.json)
  - VAMOS_CPP_COMPARE_OUT: output JSON
      (default: results/cpp_native_refactor_after/comparison_vs_baseline.json)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def _resolve_path(env_name: str, default_rel: str) -> Path:
    return Path(os.environ.get(env_name, str(ROOT_DIR / default_rel)))


def main() -> None:
    baseline_path = _resolve_path(
        "VAMOS_CPP_COMPARE_BASELINE",
        "results/cpp_native_refactor_baseline/baseline_summary.json",
    )
    after_path = _resolve_path(
        "VAMOS_CPP_COMPARE_AFTER",
        "results/cpp_native_refactor_after/after_summary.json",
    )
    out_path = _resolve_path(
        "VAMOS_CPP_COMPARE_OUT",
        "results/cpp_native_refactor_after/comparison_vs_baseline.json",
    )

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline summary not found: {baseline_path}")
    if not after_path.exists():
        raise FileNotFoundError(f"After summary not found: {after_path}")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))

    comparison: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    for key, after_payload in after.items():
        if key not in baseline:
            missing.append(key)
            continue
        b = float(baseline[key]["median_seconds"])
        a = float(after_payload["median_seconds"])
        if a <= 0.0:
            continue
        comparison[key] = {
            "baseline_median_seconds": b,
            "after_median_seconds": a,
            "speedup_x": b / a,
            "runtime_reduction_percent": ((b - a) / b * 100.0) if b > 0.0 else 0.0,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print(f"Wrote comparison: {out_path}")
    if missing:
        print(f"Warning: {len(missing)} keys from after summary were not present in baseline.")
        for key in missing:
            print(f"  - {key}")
    for key, payload in sorted(comparison.items()):
        print(
            f"{key}: baseline={payload['baseline_median_seconds']:.6f}s "
            f"after={payload['after_median_seconds']:.6f}s "
            f"speedup={payload['speedup_x']:.3f}x "
            f"reduction={payload['runtime_reduction_percent']:.2f}%"
        )


if __name__ == "__main__":
    main()
