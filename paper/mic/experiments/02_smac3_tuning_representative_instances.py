"""
MIC Experiment 02: SMAC3 tuning on representative instances.

Workflow:
1) Read representative instances selected in Experiment 01.
2) Tune NSGA-II with SMAC3 on that subset.
3) Export best configuration + top-k distinct configurations.

By default, this script enforces at least 5 distinct configurations.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vamos import optimize
from vamos.engine.tuning import (
    EvalContext,
    Instance,
    ModelBasedTuner,
    TrialResult,
    TuningTask,
    available_model_based_backends,
    build_nsgaii_config_space,
    config_from_assignment,
    filter_active_config,
    save_history_csv,
    save_history_json,
)
from vamos.foundation.metrics.hypervolume import hypervolume
from vamos.foundation.metrics.pareto import pareto_filter
from vamos.foundation.problem.registry import make_problem_selection


ROOT_DIR = Path(__file__).resolve().parents[3]
REF_DIR = ROOT_DIR / "src" / "vamos" / "foundation" / "data" / "reference_fronts"
REF_EPS = 1e-6

_PROBLEM_OVERRIDES: dict[str, dict[str, int]] = {
    # LSMOP (MIC setup): 2 objectives, 100 variables.
    **{f"lsmop{i}": {"n_obj": 2, "n_var": 100} for i in range(1, 10)},
    # C-DTLZ / DC-DTLZ: 2 objectives, 12 variables.
    **{k: {"n_obj": 2, "n_var": 12} for k in ["c1dtlz1", "c1dtlz3", "c2dtlz2", "dc1dtlz1", "dc1dtlz3", "dc2dtlz1", "dc2dtlz3"]},
    # MW: 2 objectives, 15 variables.
    **{f"mw{i}": {"n_obj": 2, "n_var": 15} for i in [1, 2, 3, 5, 6, 7]},
}


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _json_dumps(obj: Any, *, indent: int | None = None, sort_keys: bool = False) -> str:
    return json.dumps(obj, indent=indent, sort_keys=sort_keys, default=_json_default)


def _configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> list[int]:
    out: list[int] = []
    for item in _parse_csv_list(raw):
        out.append(int(item))
    if not out:
        raise ValueError("Expected at least one integer.")
    return out


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs == -1:
        return max(1, int(os.cpu_count() or 1) - 1)
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1 or -1.")
    return int(n_jobs)


def _latest_selection_json() -> Path:
    base = ROOT_DIR / "experiments" / "mic" / "instance_selection"
    candidates = sorted(base.glob("representative_instances_mic_runtime_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "No representative-instance JSON found in experiments/mic/instance_selection/. "
            "Run Experiment 01 first."
        )
    return candidates[0]


def _load_selected_instances(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("selected_instances")
    if not isinstance(raw, list):
        raise ValueError(f"Invalid selection file (missing selected_instances list): {path}")
    out = [str(x).strip().lower() for x in raw if str(x).strip()]
    if not out:
        raise ValueError(f"Selection file has no instances: {path}")
    return out


def _load_instances_from_csv(path: Path) -> list[str]:
    names: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if "problem" not in reader.fieldnames:
            raise ValueError(f"CSV file must include a 'problem' column: {path}")
        for row in reader:
            value = str(row.get("problem", "")).strip().lower()
            if value:
                names.append(value)
    if not names:
        raise ValueError(f"CSV file has no problem rows: {path}")
    return names


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = str(item).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _make_problem_selection(problem_name: str):
    overrides = _PROBLEM_OVERRIDES.get(problem_name.lower(), {})
    return make_problem_selection(problem_name, **overrides)


_REF_POINT_CACHE: dict[str, np.ndarray] = {}
_REF_HV_CACHE: dict[str, float] = {}


def _load_reference_front(problem_name: str) -> np.ndarray:
    name = problem_name.lower()
    path = REF_DIR / f"{name}.csv"
    if not path.is_file():
        alt = REF_DIR / f"{name.upper()}.csv"
        if alt.is_file():
            path = alt
        else:
            raise FileNotFoundError(f"Missing reference front for '{name}': {path}")
    return np.loadtxt(path, delimiter=",")


def _reference_point(problem_name: str) -> np.ndarray:
    key = problem_name.lower()
    if key in _REF_POINT_CACHE:
        return _REF_POINT_CACHE[key]
    front = _load_reference_front(key)
    ref = front.max(axis=0) + REF_EPS
    _REF_POINT_CACHE[key] = ref
    return ref


def _reference_hv(problem_name: str) -> float:
    key = problem_name.lower()
    if key in _REF_HV_CACHE:
        return _REF_HV_CACHE[key]
    front = _load_reference_front(key)
    ref = _reference_point(key)
    front = front[np.all(front <= ref, axis=1)]
    hv = float(hypervolume(front, ref, allow_ref_expand=False)) if front.size else 0.0
    _REF_HV_CACHE[key] = hv
    return hv


def compute_normalized_hv(F: Any, problem_name: str) -> float:
    if F is None:
        return 0.0
    arr = np.asarray(F, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0
    front = pareto_filter(arr)
    if front is None or front.size == 0:
        return 0.0
    ref = _reference_point(problem_name)
    front = front[np.all(front <= ref, axis=1)]
    if front.size == 0:
        return 0.0
    hv = float(hypervolume(front, ref, allow_ref_expand=False))
    hv_ref = _reference_hv(problem_name)
    return float(hv / hv_ref) if hv_ref > 0 else 0.0


def make_evaluator(*, engine: str, failure_score: float):
    def eval_fn(config: dict[str, Any], ctx: EvalContext) -> float:
        problem_name = str(ctx.instance.name).strip().lower()
        kwargs = dict(getattr(ctx.instance, "kwargs", {}) or {})
        try:
            selection = make_problem_selection(problem_name, **kwargs)
            problem = selection.instantiate()
            algo_cfg = config_from_assignment("nsgaii", dict(config))
            result = optimize(
                problem,
                algorithm="nsgaii",
                algorithm_config=algo_cfg,
                termination=("max_evaluations", int(ctx.budget)),
                seed=int(ctx.seed),
                engine=str(engine),
            )
            score = compute_normalized_hv(getattr(result, "F", None), problem_name)
            if not np.isfinite(score):
                return float(failure_score)
            return float(score)
        except Exception as exc:
            _logger().warning("Evaluation failed for %s (seed=%s, budget=%s): %s", problem_name, ctx.seed, ctx.budget, exc)
            return float(failure_score)

    return eval_fn


def _rank_unique_configs(
    *,
    history: list[TrialResult],
    task: TuningTask,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    best_by_key: dict[str, dict[str, Any]] = {}
    for tr in history:
        active_cfg = filter_active_config(dict(tr.config), task.param_space)
        key = _json_dumps(active_cfg, sort_keys=True)
        row = best_by_key.get(key)
        score = float(tr.score)
        if row is None or score > float(row["score"]):
            best_by_key[key] = {
                "score": score,
                "config": active_cfg,
                "trial_id": int(tr.trial_id),
                "details": dict(tr.details),
            }
    ranked = sorted(best_by_key.values(), key=lambda x: float(x["score"]), reverse=True)
    return ranked[: max(1, int(top_k))], ranked


def _resolve_output_dir(base: Path, name: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = name.strip() if name.strip() else f"smac3_rep_instances_{ts}"
    out = base.expanduser().resolve() / suffix
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_top_configs_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    simple_rows: list[dict[str, Any]] = []
    for row in rows:
        simple_rows.append(
            {
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "trial_id": int(row["trial_id"]),
                "config_json": _json_dumps(row["config"], sort_keys=True),
                "resolved_config_json": _json_dumps(row["resolved_config"], sort_keys=True),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rank", "score", "trial_id", "config_json", "resolved_config_json"])
        writer.writeheader()
        writer.writerows(simple_rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MIC Experiment 02: SMAC3 tuning on representative instances.")
    ap.add_argument(
        "--selection-file",
        type=str,
        default="",
        help="Selection file from Experiment 01 (.json or .csv). Default: latest JSON in experiments/mic/instance_selection.",
    )
    ap.add_argument(
        "--instances",
        type=str,
        default="",
        help="Optional comma-separated instance list. If provided, overrides --selection-file.",
    )
    ap.add_argument("--budget", type=int, default=20000, help="Evaluation budget per run.")
    ap.add_argument(
        "--budget-levels",
        type=str,
        default="5000,10000,20000",
        help="Comma-separated multi-fidelity levels for model-based tuner.",
    )
    ap.add_argument("--max-trials", type=int, default=120, help="Total SMAC3 trials.")
    ap.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds for each config evaluation.")
    ap.add_argument("--seed", type=int, default=42, help="Global tuner seed.")
    ap.add_argument("--engine", type=str, default="numba", help="Execution backend for optimize().")
    ap.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers (-1 = CPU cores - 1).")
    ap.add_argument(
        "--timeout-seconds",
        type=float,
        default=32400.0,
        help="Wallclock timeout in seconds (default: 32400 = 9h). Set <= 0 to disable.",
    )
    ap.add_argument("--show-progress-bar", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--top-k", type=int, default=5, help="Number of top distinct configs to export.")
    ap.add_argument("--min-distinct", type=int, default=5, help="Minimum number of distinct configs required.")
    ap.add_argument(
        "--strict-min-distinct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, fail when fewer than --min-distinct distinct configs are found.",
    )
    ap.add_argument("--failure-score", type=float, default=0.0, help="Score assigned on failed evaluations.")
    ap.add_argument("--output-dir", type=Path, default=Path("experiments") / "mic" / "smac3_tuning")
    ap.add_argument("--name", type=str, default="", help="Optional output subfolder name.")
    ap.add_argument("--dry-run", action="store_true", help="Print resolved setup and exit.")
    return ap.parse_args()


def main() -> None:
    _configure_logging()
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0.")
    if args.min_distinct <= 0:
        raise ValueError("--min-distinct must be > 0.")
    if args.max_trials <= 0:
        raise ValueError("--max-trials must be > 0.")
    if args.budget <= 0:
        raise ValueError("--budget must be > 0.")
    if args.max_trials < args.min_distinct and args.strict_min_distinct:
        raise ValueError(
            f"--max-trials ({args.max_trials}) is smaller than --min-distinct ({args.min_distinct}); "
            "cannot guarantee the requested number of distinct configurations."
        )

    if args.instances.strip():
        selected_instances = _dedupe_keep_order([x.lower() for x in _parse_csv_list(args.instances)])
        selection_source = "cli --instances"
    else:
        selection_path = Path(args.selection_file).resolve() if args.selection_file.strip() else _latest_selection_json()
        if selection_path.suffix.lower() == ".json":
            selected_instances = _dedupe_keep_order(_load_selected_instances(selection_path))
        elif selection_path.suffix.lower() == ".csv":
            selected_instances = _dedupe_keep_order(_load_instances_from_csv(selection_path))
        else:
            raise ValueError(f"Unsupported selection-file extension: {selection_path.suffix}")
        selection_source = str(selection_path)

    if not selected_instances:
        raise RuntimeError("No selected instances provided.")

    seeds = _parse_csv_ints(args.seeds)
    budget_levels = _parse_csv_ints(args.budget_levels)
    budget_levels = sorted(set(min(int(args.budget), max(1, int(x))) for x in budget_levels))
    if budget_levels[-1] != int(args.budget):
        budget_levels.append(int(args.budget))
    n_jobs = _resolve_n_jobs(int(args.n_jobs))

    instances: list[Instance] = []
    instance_meta: list[dict[str, Any]] = []
    for problem_name in selected_instances:
        overrides = dict(_PROBLEM_OVERRIDES.get(problem_name, {}))
        sel = make_problem_selection(problem_name, **overrides)
        instances.append(Instance(name=problem_name, n_var=int(sel.n_var), kwargs=overrides))
        instance_meta.append(
            {
                "problem": problem_name,
                "n_var": int(sel.n_var),
                "n_obj": int(sel.n_obj),
                "kwargs": overrides,
            }
        )

    _logger().info("Selected instances (%s): %s", len(selected_instances), ", ".join(selected_instances))
    _logger().info("Selection source: %s", selection_source)
    _logger().info("Budget: %s", args.budget)
    _logger().info("Budget levels: %s", budget_levels)
    _logger().info("Trials: %s", args.max_trials)
    _logger().info("Seeds: %s", seeds)
    _logger().info("Timeout (seconds): %s", args.timeout_seconds)
    _logger().info("Top-k distinct: %s (min required: %s)", args.top_k, args.min_distinct)
    _logger().info("Resolved n_jobs: %s", n_jobs)
    available = available_model_based_backends()
    smac_available = bool(available.get("smac3", False))
    _logger().info("SMAC3 backend available: %s", smac_available)

    if args.dry_run:
        return

    if not smac_available:
        raise RuntimeError(
            "Backend 'smac3' is unavailable. Install optional tuning dependencies first."
        )

    param_space = build_nsgaii_config_space().to_param_space()
    task = TuningTask(
        name="mic_smac3_nsgaii_representative_instances",
        param_space=param_space,
        instances=instances,
        seeds=seeds,
        budget_per_run=int(args.budget),
        maximize=True,
        aggregator=np.mean,
    )
    eval_fn = make_evaluator(engine=str(args.engine), failure_score=float(args.failure_score))

    out_dir = _resolve_output_dir(Path(args.output_dir), str(args.name))
    _logger().info("Output directory: %s", out_dir)

    tuner = ModelBasedTuner(
        task=task,
        max_trials=int(args.max_trials),
        backend="smac3",
        seed=int(args.seed),
        n_jobs=int(n_jobs),
        timeout_seconds=None if float(args.timeout_seconds) <= 0.0 else float(args.timeout_seconds),
        show_progress_bar=bool(args.show_progress_bar),
        budget_levels=list(budget_levels),
    )

    t0 = time.perf_counter()
    best_config, history = tuner.run(eval_fn, verbose=True)
    elapsed_s = float(time.perf_counter() - t0)

    save_history_json(history, task.param_space, out_dir / "tuning_history.json", include_raw=True)
    save_history_csv(history, task.param_space, out_dir / "tuning_history.csv", include_raw=True)

    best_active = filter_active_config(dict(best_config), task.param_space)
    best_config_path = out_dir / "best_config_raw.json"
    best_active_path = out_dir / "best_config_active.json"
    best_resolved_path = out_dir / "best_config_resolved.json"
    best_config_path.write_text(_json_dumps(best_config, indent=2), encoding="utf-8")
    best_active_path.write_text(_json_dumps(best_active, indent=2), encoding="utf-8")
    best_resolved = config_from_assignment("nsgaii", best_active).to_dict()
    best_resolved_path.write_text(_json_dumps(best_resolved, indent=2), encoding="utf-8")

    top_ranked, all_ranked = _rank_unique_configs(history=history, task=task, top_k=int(args.top_k))
    n_distinct = len(all_ranked)
    if n_distinct < int(args.min_distinct):
        message = (
            f"Only {n_distinct} distinct configurations found, below min-distinct={args.min_distinct}. "
            "Increase --max-trials or widen the search space."
        )
        if args.strict_min_distinct:
            raise RuntimeError(message)
        _logger().warning(message)

    top_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(top_ranked, start=1):
        cfg = dict(row["config"])
        top_rows.append(
            {
                "rank": rank,
                "score": float(row["score"]),
                "trial_id": int(row["trial_id"]),
                "config": cfg,
                "resolved_config": config_from_assignment("nsgaii", cfg).to_dict(),
                "details": dict(row["details"]),
            }
        )

    top_json_path = out_dir / "top_configs_distinct.json"
    top_csv_path = out_dir / "top_configs_distinct.csv"
    top_json_path.write_text(_json_dumps(top_rows, indent=2), encoding="utf-8")
    _write_top_configs_csv(top_csv_path, top_rows)

    selected_payload = {
        "selection_source": selection_source,
        "selected_instances": selected_instances,
        "instance_meta": instance_meta,
    }
    (out_dir / "selected_instances.json").write_text(_json_dumps(selected_payload, indent=2), encoding="utf-8")

    summary = {
        "schema_version": "mic_smac3_tuning_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "smac3",
        "engine": str(args.engine),
        "budget": int(args.budget),
        "budget_levels": budget_levels,
        "max_trials": int(args.max_trials),
        "seeds": seeds,
        "timeout_seconds": None if float(args.timeout_seconds) <= 0.0 else float(args.timeout_seconds),
        "n_jobs_resolved": int(n_jobs),
        "selected_instances_count": len(selected_instances),
        "history_rows": len(history),
        "distinct_configs_found": int(n_distinct),
        "top_k_requested": int(args.top_k),
        "min_distinct_requested": int(args.min_distinct),
        "strict_min_distinct": bool(args.strict_min_distinct),
        "elapsed_seconds": elapsed_s,
        "best_score": float(max((h.score for h in history), default=float("nan"))),
        "artifacts": {
            "selected_instances": "selected_instances.json",
            "history_json": "tuning_history.json",
            "history_csv": "tuning_history.csv",
            "best_config_raw": "best_config_raw.json",
            "best_config_active": "best_config_active.json",
            "best_config_resolved": "best_config_resolved.json",
            "top_configs_distinct_json": "top_configs_distinct.json",
            "top_configs_distinct_csv": "top_configs_distinct.csv",
        },
    }
    (out_dir / "tuning_summary.json").write_text(_json_dumps(summary, indent=2), encoding="utf-8")

    _logger().info("Done. Distinct configs found: %s", n_distinct)
    _logger().info("Top distinct exported: %s", len(top_rows))
    _logger().info("Summary: %s", out_dir / "tuning_summary.json")


if __name__ == "__main__":
    main()
