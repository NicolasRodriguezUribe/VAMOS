"""
Run native C++ runtime benchmarks for NSGA-II, SMS-EMOA, and SPEA2.

This script is designed to reproduce "before/after" runtime comparisons in the
same artifact format used for the NSGA-II native-path refactor:
  - baseline_summary.json / after_summary.json
  - baseline_cprofile.txt / after_cprofile.txt
  - comparison_vs_baseline.json (when baseline is available)

Usage:
  python paper/30_run_cpp_native_refactor_benchmark.py

Environment variables:
  - VAMOS_CPP_BENCH_PHASE: baseline|after (default: after)
  - VAMOS_CPP_BENCH_ALGORITHMS: comma list (default: nsgaii,smsemoa,spea2)
  - VAMOS_CPP_BENCH_PROBLEM: problem name (default: zdt1)
  - VAMOS_CPP_BENCH_CASES: e<evals>_p<pop>;... (default: e20000_p100;e50000_p500)
  - VAMOS_CPP_BENCH_SEEDS_SMALL: comma seeds for first case (default: 11,22,33)
  - VAMOS_CPP_BENCH_SEEDS_LARGE: comma seeds for second case (default: 1,2,3)
  - VAMOS_CPP_BENCH_OUTPUT_DIR: output dir
      (default: results/cpp_native_refactor_<phase>)
  - VAMOS_CPP_BENCH_BASELINE_SUMMARY: baseline summary for comparison
      (default: results/cpp_native_refactor_baseline/baseline_summary.json)
  - VAMOS_CPP_BENCH_PROFILE: 1/0 write cProfile capture (default: 1)
"""

from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT_DIR / "src"))

from vamos import optimize


@dataclass(frozen=True)
class BenchCase:
    label: str
    max_evaluations: int
    pop_size: int
    seeds: tuple[int, ...]


def _parse_int_list(raw: str) -> tuple[int, ...]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one integer value.")
    return tuple(vals)


def _parse_case_list(raw: str) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for item in raw.split(";"):
        token = item.strip().lower()
        if not token:
            continue
        if not token.startswith("e") or "_p" not in token:
            raise ValueError(f"Invalid case token '{item}'. Expected format e<evals>_p<pop>.")
        eval_part, pop_part = token.split("_p", maxsplit=1)
        out.append((int(eval_part[1:]), int(pop_part)))
    if not out:
        raise ValueError("No benchmark cases parsed from VAMOS_CPP_BENCH_CASES.")
    return tuple(out)


def _summary_filename(phase: str) -> str:
    if phase == "baseline":
        return "baseline_summary.json"
    if phase == "after":
        return "after_summary.json"
    return f"{phase}_summary.json"


def _cprofile_filename(phase: str) -> str:
    if phase == "baseline":
        return "baseline_cprofile.txt"
    if phase == "after":
        return "after_cprofile.txt"
    return f"{phase}_cprofile.txt"


def _resolve_cases() -> tuple[BenchCase, ...]:
    raw_cases = os.environ.get("VAMOS_CPP_BENCH_CASES", "e20000_p100;e50000_p500")
    parsed_cases = _parse_case_list(raw_cases)

    seeds_small = _parse_int_list(os.environ.get("VAMOS_CPP_BENCH_SEEDS_SMALL", "11,22,33"))
    seeds_large = _parse_int_list(os.environ.get("VAMOS_CPP_BENCH_SEEDS_LARGE", "1,2,3"))
    default_by_index = {0: seeds_small, 1: seeds_large}

    cases: list[BenchCase] = []
    for i, (n_evals, pop) in enumerate(parsed_cases):
        seeds = default_by_index.get(i, seeds_large)
        cases.append(BenchCase(label=f"e{n_evals}_p{pop}", max_evaluations=n_evals, pop_size=pop, seeds=seeds))
    return tuple(cases)


def _normalize_algorithms(raw: str) -> tuple[str, ...]:
    supported = {"nsgaii", "smsemoa", "spea2"}
    vals = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("VAMOS_CPP_BENCH_ALGORITHMS is empty.")
    unknown = sorted(set(vals) - supported)
    if unknown:
        raise ValueError(f"Unsupported algorithms: {unknown}. Supported: {sorted(supported)}")
    dedup: list[str] = []
    seen: set[str] = set()
    for val in vals:
        if val not in seen:
            dedup.append(val)
            seen.add(val)
    return tuple(dedup)


def _benchmark_one_case(algorithm: str, problem: str, case: BenchCase) -> dict[str, object]:
    times: list[float] = []
    for seed in case.seeds:
        kwargs = {
            "problem": problem,
            "algorithm": algorithm,
            "engine": "cpp",
            "max_evaluations": int(case.max_evaluations),
            "pop_size": int(case.pop_size),
            "seed": int(seed),
        }
        start = time.perf_counter()
        optimize(**kwargs)
        times.append(time.perf_counter() - start)
    return {
        "algorithm": algorithm,
        "problem": problem,
        "max_evaluations": case.max_evaluations,
        "pop_size": case.pop_size,
        "seeds": list(case.seeds),
        "times_seconds": times,
        "median_seconds": statistics.median(times),
        "mean_seconds": statistics.mean(times),
    }


def _capture_profile(algorithm: str, problem: str, case: BenchCase) -> str:
    run_kwargs = {
        "problem": problem,
        "algorithm": algorithm,
        "engine": "cpp",
        "max_evaluations": int(case.max_evaluations),
        "pop_size": int(case.pop_size),
        "seed": int(case.seeds[0]),
    }
    profile = cProfile.Profile()
    profile.enable()
    optimize(**run_kwargs)
    profile.disable()
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).sort_stats("cumtime").print_stats(120)
    return stream.getvalue()


def _make_comparison(
    *,
    baseline_path: Path,
    after_summary: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    if not baseline_path.exists():
        return
    baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))
    comparison: dict[str, dict[str, float]] = {}
    for key, after_payload in after_summary.items():
        if key not in baseline_summary:
            continue
        base_median = float(baseline_summary[key]["median_seconds"])
        after_median = float(after_payload["median_seconds"])
        if after_median <= 0.0:
            continue
        comparison[key] = {
            "baseline_median_seconds": base_median,
            "after_median_seconds": after_median,
            "speedup_x": base_median / after_median,
            "runtime_reduction_percent": (base_median - after_median) / base_median * 100.0 if base_median > 0.0 else 0.0,
        }
    if comparison:
        output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")


def _prepare_windows_dll_dirs() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    candidates = [
        os.environ.get("VAMOSPP_DLL_DIR"),
        str(Path(sys.executable).resolve().parent),
    ]
    for tool in ("g++", "gcc", "clang++"):
        tool_path = shutil.which(tool)
        if tool_path:
            candidates.append(str(Path(tool_path).resolve().parent))

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            os.add_dll_directory(str(path))
        except (FileNotFoundError, OSError):
            pass


def main() -> None:
    phase = os.environ.get("VAMOS_CPP_BENCH_PHASE", "after").strip().lower()
    algorithms = _normalize_algorithms(os.environ.get("VAMOS_CPP_BENCH_ALGORITHMS", "nsgaii,smsemoa,spea2"))
    problem = os.environ.get("VAMOS_CPP_BENCH_PROBLEM", "zdt1").strip().lower()
    cases = _resolve_cases()

    out_dir_default = ROOT_DIR / "results" / f"cpp_native_refactor_{phase}"
    out_dir = Path(os.environ.get("VAMOS_CPP_BENCH_OUTPUT_DIR", str(out_dir_default)))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep the current flags unchanged and enabled for native evolve path.
    os.environ["VAMOS_ENABLE_CPP_EVOLVE_FASTPATH"] = "1"
    os.environ["VAMOS_ENABLE_CPP_NATIVE_EVOLVE"] = "1"

    _prepare_windows_dll_dirs()
    import vamospp

    if not vamospp.is_native_backend():
        raise RuntimeError("vamospp native backend is not active. Install VAMOS++ before running this benchmark.")

    print("CPP native benchmark configuration")
    print(f"- phase: {phase}")
    print(f"- algorithms: {list(algorithms)}")
    print(f"- problem: {problem}")
    print(f"- cases: {[c.label for c in cases]}")
    print(f"- output_dir: {out_dir}")

    summary: dict[str, dict[str, object]] = {}
    for algorithm in algorithms:
        for case in cases:
            key = f"{algorithm}_{problem}_{case.label}"
            payload = _benchmark_one_case(algorithm=algorithm, problem=problem, case=case)
            summary[key] = payload
            print(f"  {key}: median={payload['median_seconds']:.6f}s")

    summary_path = out_dir / _summary_filename(phase)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if int(os.environ.get("VAMOS_CPP_BENCH_PROFILE", "1")):
        profile_blobs: list[str] = []
        for algorithm in algorithms:
            profile_blobs.append(f"\n=== {algorithm} ({problem}, {cases[0].label}) ===\n")
            profile_blobs.append(_capture_profile(algorithm=algorithm, problem=problem, case=cases[0]))
        (out_dir / _cprofile_filename(phase)).write_text("".join(profile_blobs), encoding="utf-8")

    baseline_default = ROOT_DIR / "results" / "cpp_native_refactor_baseline" / "baseline_summary.json"
    baseline_path = Path(os.environ.get("VAMOS_CPP_BENCH_BASELINE_SUMMARY", str(baseline_default)))
    if phase == "after":
        _make_comparison(
            baseline_path=baseline_path,
            after_summary=summary,
            output_path=out_dir / "comparison_vs_baseline.json",
        )

    print(f"Wrote summary: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
