"""
Shared helpers for the 3-objective ZCAT difficulty robustness campaign.

This campaign extends the canonical 3-objective farthest/GCES survival-only
setup with official ZCAT difficulty controls:
- level
- bias
- imbalance
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Sequence

import numpy as np

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from vamos import optimize
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.foundation.problem import zcat as zcat_problems

try:
    from .benchmark_utils import compute_hv, compute_igd_plus
    from .gces_ablation_zcat_2obj_common import (
        ALGORITHM_LABELS as SHARED_ALGORITHM_LABELS,
        DEFAULT_ENGINE,
        DEFAULT_EVALS,
        DEFAULT_N_VAR,
        DEFAULT_POP_SIZE,
        DEFAULT_PROBLEMS,
        build_config,
        holm_adjust,
        metric_problem_name,
        paired_wilcoxon_pvalue,
        write_csv,
        write_run_config,
    )
except ImportError:
    from benchmark_utils import compute_hv, compute_igd_plus
    from gces_ablation_zcat_2obj_common import (
        ALGORITHM_LABELS as SHARED_ALGORITHM_LABELS,
        DEFAULT_ENGINE,
        DEFAULT_EVALS,
        DEFAULT_N_VAR,
        DEFAULT_POP_SIZE,
        DEFAULT_PROBLEMS,
        build_config,
        holm_adjust,
        metric_problem_name,
        paired_wilcoxon_pvalue,
        write_csv,
        write_run_config,
    )


ALGORITHM_LABELS = dict(SHARED_ALGORITHM_LABELS)
DEFAULT_ALGORITHMS = [
    "nsgaii",
    "nsga2_farthest",
    "nsga2_hvref_farthest",
    "nsga2_hvfarthest",
    "gces_nogeo",
    "gces",
]
DEFAULT_COMPARISONS = [
    ("nsga2_farthest", "nsgaii"),
    ("nsga2_hvref_farthest", "nsgaii"),
    ("nsga2_hvfarthest", "nsgaii"),
    ("gces_nogeo", "nsgaii"),
    ("gces", "nsgaii"),
]
DEFAULT_SEEDS = 21
DEFAULT_N_OBJ = 3
DEFAULT_MAX_WORKERS = 18
DEFAULT_OUTPUT_DIR = ROOT_DIR / "experiments" / "farthest_gces_difficulty_ab_zcat_3obj_full"


@dataclass(frozen=True)
class DifficultyCell:
    config_id: str
    config_label: str
    phase: str
    level: int
    bias: bool
    imbalance: bool


DEFAULT_DIFFICULTY_CELLS: tuple[DifficultyCell, ...] = (
    DifficultyCell("lvl1_plain", "L1", "A", 1, False, False),
    DifficultyCell("lvl3_plain", "L3", "A", 3, False, False),
    DifficultyCell("lvl6_plain", "L6", "A", 6, False, False),
    DifficultyCell("lvl6_bias", "L6+B", "B", 6, True, False),
    DifficultyCell("lvl6_imbalance", "L6+I", "B", 6, False, True),
    DifficultyCell("lvl6_bias_imbalance", "L6+B+I", "B", 6, True, True),
)


@dataclass(frozen=True)
class RunRecord:
    config_id: str
    config_label: str
    phase: str
    level: int
    bias: bool
    imbalance: bool
    problem: str
    algorithm: str
    algorithm_label: str
    seed: int
    engine: str
    pop_size: int
    max_evaluations: int
    n_var: int
    n_obj: int
    runtime_seconds: float
    n_solutions: int
    hypervolume: float
    igd_plus: float


def _float_or_zero(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(stdev(values))


def _median_or_zero(values: list[float]) -> float:
    return 0.0 if not values else float(median(values))


def _format_number(value: float | None, *, digits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    separator = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(separator) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _coerce_comparisons(comparisons: Sequence[tuple[str, str]] | Sequence[Sequence[str]]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for pair in comparisons:
        if len(pair) != 2:
            raise ValueError(f"Invalid comparison pair: {pair!r}")
        pairs.append((str(pair[0]), str(pair[1])))
    return pairs


def _bool_from_str(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def resolve_worker_count(workers: int | None) -> int:
    available = max(1, int(os.cpu_count() or 1))
    if workers is None:
        return max(1, min(DEFAULT_MAX_WORKERS, available))
    return max(1, min(int(workers), available))


def select_difficulty_cells(config_ids: Sequence[str] | None = None) -> list[DifficultyCell]:
    if config_ids is None:
        return list(DEFAULT_DIFFICULTY_CELLS)
    lookup = {cell.config_id: cell for cell in DEFAULT_DIFFICULTY_CELLS}
    selected: list[DifficultyCell] = []
    for config_id in config_ids:
        key = str(config_id).strip()
        if key not in lookup:
            available = ", ".join(cell.config_id for cell in DEFAULT_DIFFICULTY_CELLS)
            raise ValueError(f"Unknown config_id '{config_id}'. Use one of: {available}")
        selected.append(lookup[key])
    return selected


def instantiate_zcat_problem(
    problem: str,
    *,
    n_var: int,
    n_obj: int,
    level: int,
    bias: bool,
    imbalance: bool,
) -> Any:
    key = str(problem).strip().lower()
    if not key.startswith("zcat"):
        raise ValueError(f"Unsupported difficulty campaign problem '{problem}'. Expected a ZCAT problem key.")
    try:
        index = int(key.removeprefix("zcat"))
    except ValueError as exc:
        raise ValueError(f"Invalid ZCAT problem key '{problem}'.") from exc
    cls_name = f"ZCAT{index}Problem"
    try:
        problem_cls = getattr(zcat_problems, cls_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported ZCAT problem key '{problem}'.") from exc
    return problem_cls(n_var=int(n_var), n_obj=int(n_obj), level=int(level), bias=bool(bias), imbalance=bool(imbalance))


def run_once(
    *,
    cell: DifficultyCell,
    problem: str,
    algorithm: str,
    seed: int,
    engine: str,
    pop_size: int,
    max_evaluations: int,
    n_var: int,
    n_obj: int,
    algorithm_config: NSGAIIConfig | None = None,
) -> RunRecord:
    config = algorithm_config if algorithm_config is not None else build_config(pop_size=pop_size, n_var=n_var)
    problem_instance = instantiate_zcat_problem(
        problem,
        n_var=n_var,
        n_obj=n_obj,
        level=cell.level,
        bias=cell.bias,
        imbalance=cell.imbalance,
    )
    start = time.perf_counter()
    result = optimize(
        problem_instance,
        algorithm=algorithm,
        algorithm_config=config,
        max_evaluations=max_evaluations,
        seed=seed,
        engine=engine,
        n_var=n_var,
        n_obj=n_obj,
    )
    elapsed = time.perf_counter() - start
    F = result.F
    metric_problem = metric_problem_name(problem, n_obj)
    return RunRecord(
        config_id=cell.config_id,
        config_label=cell.config_label,
        phase=cell.phase,
        level=int(cell.level),
        bias=bool(cell.bias),
        imbalance=bool(cell.imbalance),
        problem=problem,
        algorithm=algorithm,
        algorithm_label=ALGORITHM_LABELS[algorithm],
        seed=int(seed),
        engine=str(engine),
        pop_size=int(pop_size),
        max_evaluations=int(max_evaluations),
        n_var=int(n_var),
        n_obj=int(n_obj),
        runtime_seconds=float(elapsed),
        n_solutions=int(F.shape[0]),
        hypervolume=float(compute_hv(F, metric_problem)),
        igd_plus=float(compute_igd_plus(F, metric_problem)),
    )


def _task_order_key(task: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(task["config_index"]),
        int(task["problem_index"]),
        int(task["algorithm_index"]),
        int(task["seed"]),
    )


def _run_task(task: dict[str, Any]) -> tuple[tuple[int, int, int, int], RunRecord]:
    order_key = _task_order_key(task)
    cell = DifficultyCell(
        config_id=str(task["config_id"]),
        config_label=str(task["config_label"]),
        phase=str(task["phase"]),
        level=int(task["level"]),
        bias=bool(task["bias"]),
        imbalance=bool(task["imbalance"]),
    )
    record = run_once(
        cell=cell,
        problem=str(task["problem"]),
        algorithm=str(task["algorithm"]),
        seed=int(task["seed"]),
        engine=str(task["engine"]),
        pop_size=int(task["pop_size"]),
        max_evaluations=int(task["max_evaluations"]),
        n_var=int(task["n_var"]),
        n_obj=int(task["n_obj"]),
        algorithm_config=None,
    )
    return order_key, record


def _format_progress_line(record: RunRecord, completed: int, total: int) -> str:
    return (
        f"[{completed:>5}/{total}] "
        f"{record.config_label:>6} | "
        f"{record.problem:<7} | "
        f"{record.algorithm_label:>24} seed={record.seed:>2} | "
        f"HV={record.hypervolume:.4f} | "
        f"IGD+={record.igd_plus:.4f} | "
        f"time={record.runtime_seconds:.2f}s"
    )


def run_campaign(
    *,
    cells: Sequence[DifficultyCell],
    problems: Sequence[str],
    algorithms: Sequence[str],
    seeds: Sequence[int],
    engine: str,
    pop_size: int,
    max_evaluations: int,
    n_var: int,
    n_obj: int,
    workers: int | None = None,
    show_progress: bool = True,
) -> list[RunRecord]:
    resolved_workers = resolve_worker_count(workers)
    tasks: list[dict[str, Any]] = []
    for config_index, cell in enumerate(cells):
        for problem_index, problem in enumerate(problems):
            for algorithm_index, algorithm in enumerate(algorithms):
                for seed in seeds:
                    tasks.append(
                        {
                            "config_id": cell.config_id,
                            "config_label": cell.config_label,
                            "phase": cell.phase,
                            "level": int(cell.level),
                            "bias": bool(cell.bias),
                            "imbalance": bool(cell.imbalance),
                            "problem": str(problem),
                            "algorithm": str(algorithm),
                            "seed": int(seed),
                            "engine": str(engine),
                            "pop_size": int(pop_size),
                            "max_evaluations": int(max_evaluations),
                            "n_var": int(n_var),
                            "n_obj": int(n_obj),
                            "config_index": int(config_index),
                            "problem_index": int(problem_index),
                            "algorithm_index": int(algorithm_index),
                        }
                    )

    indexed_records: list[tuple[tuple[int, int, int, int], RunRecord]] = []
    completed = 0
    total = len(tasks)
    if resolved_workers <= 1:
        for task in tasks:
            order_key, record = _run_task(task)
            indexed_records.append((order_key, record))
            completed += 1
            if show_progress:
                print(_format_progress_line(record, completed, total), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            future_to_task = {executor.submit(_run_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    order_key, record = future.result()
                except Exception as exc:  # pragma: no cover - surfaced in campaign logs
                    raise RuntimeError(
                        "Difficulty campaign run failed for "
                        f"config_id={task['config_id']!r}, problem={task['problem']!r}, "
                        f"algorithm={task['algorithm']!r}, seed={task['seed']!r}."
                    ) from exc
                indexed_records.append((order_key, record))
                completed += 1
                if show_progress:
                    print(_format_progress_line(record, completed, total), flush=True)

    indexed_records.sort(key=lambda item: item[0])
    return [record for _, record in indexed_records]


def load_raw_records(path: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            records.append(
                RunRecord(
                    config_id=row["config_id"],
                    config_label=row["config_label"],
                    phase=row["phase"],
                    level=int(row["level"]),
                    bias=_bool_from_str(row["bias"]),
                    imbalance=_bool_from_str(row["imbalance"]),
                    problem=row["problem"],
                    algorithm=row["algorithm"],
                    algorithm_label=row.get("algorithm_label") or ALGORITHM_LABELS[row["algorithm"]],
                    seed=int(row["seed"]),
                    engine=row["engine"],
                    pop_size=int(row["pop_size"]),
                    max_evaluations=int(row["max_evaluations"]),
                    n_var=int(row["n_var"]),
                    n_obj=int(row["n_obj"]),
                    runtime_seconds=float(row["runtime_seconds"]),
                    n_solutions=int(row["n_solutions"]),
                    hypervolume=float(row["hypervolume"]),
                    igd_plus=float(row["igd_plus"]),
                )
            )
    return records


def _win_tie_loss(
    lhs_values: list[float],
    rhs_values: list[float],
    *,
    higher_is_better: bool,
    tie_tol: float,
) -> tuple[int, int, int]:
    wins = 0
    ties = 0
    losses = 0
    for lhs_val, rhs_val in zip(lhs_values, rhs_values, strict=True):
        diff = float(lhs_val - rhs_val)
        if abs(diff) <= tie_tol:
            ties += 1
        elif (diff > 0.0 and higher_is_better) or (diff < 0.0 and not higher_is_better):
            wins += 1
        else:
            losses += 1
    return wins, ties, losses


def _build_metric_row(
    *,
    cell: DifficultyCell,
    problem: str,
    lhs_algorithm: str,
    rhs_algorithm: str,
    lhs_label: str,
    rhs_label: str,
    metric: str,
    higher_is_better: bool,
    lhs_values: list[float],
    rhs_values: list[float],
    tie_tol: float,
) -> dict[str, Any]:
    wins, ties, losses = _win_tie_loss(lhs_values, rhs_values, higher_is_better=higher_is_better, tie_tol=tie_tol)
    return {
        "config_id": cell.config_id,
        "config_label": cell.config_label,
        "phase": cell.phase,
        "level": int(cell.level),
        "bias": bool(cell.bias),
        "imbalance": bool(cell.imbalance),
        "problem": problem,
        "lhs_algorithm": lhs_algorithm,
        "rhs_algorithm": rhs_algorithm,
        "lhs_label": lhs_label,
        "rhs_label": rhs_label,
        "metric": metric,
        "lhs_mean": float(mean(lhs_values)),
        "lhs_median": _median_or_zero(lhs_values),
        "rhs_mean": float(mean(rhs_values)),
        "rhs_median": _median_or_zero(rhs_values),
        "delta_mean": float(mean(lhs_values) - mean(rhs_values)),
        "delta_median": _median_or_zero(lhs_values) - _median_or_zero(rhs_values),
        "lhs_seed_wins": int(wins),
        "seed_ties": int(ties),
        "rhs_seed_wins": int(losses),
        "wilcoxon_p_value_raw": paired_wilcoxon_pvalue(lhs_values, rhs_values),
        "wilcoxon_p_value_holm": None,
        "wilcoxon_significant_alpha": False,
    }


def _summary_lookup(summary_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(row["config_id"], row["problem"], row["algorithm"]): row for row in summary_rows}


def _global_summary_lookup(global_summary_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["config_id"], row["algorithm"]): row for row in global_summary_rows}


def summarize_records(
    records: list[RunRecord],
    *,
    comparisons: Sequence[tuple[str, str]] = DEFAULT_COMPARISONS,
    algorithm_order: Sequence[str] | None = None,
    alpha: float = 0.05,
    tie_tol: float = 1e-12,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    grouped: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
    config_lookup: dict[str, DifficultyCell] = {}
    for record in records:
        grouped[(record.config_id, record.problem, record.algorithm)].append(record)
        config_lookup.setdefault(
            record.config_id,
            DifficultyCell(record.config_id, record.config_label, record.phase, record.level, record.bias, record.imbalance),
        )

    ordered_algorithms = list(DEFAULT_ALGORITHMS if algorithm_order is None else algorithm_order)
    algorithms = [algorithm for algorithm in ordered_algorithms if any(record.algorithm == algorithm for record in records)]
    problems = sorted({record.problem for record in records})
    cells = [config_lookup[cell.config_id] for cell in DEFAULT_DIFFICULTY_CELLS if cell.config_id in config_lookup]

    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for cell in cells:
        for problem in problems:
            for algorithm in algorithms:
                runs = grouped.get((cell.config_id, problem, algorithm), [])
                if not runs:
                    continue
                hv_values = [run.hypervolume for run in runs]
                igd_values = [run.igd_plus for run in runs]
                runtime_values = [run.runtime_seconds for run in runs]
                hv_mean, hv_std = _float_or_zero(hv_values)
                igd_mean, igd_std = _float_or_zero(igd_values)
                runtime_mean, runtime_std = _float_or_zero(runtime_values)
                summary_rows.append(
                    {
                        "config_id": cell.config_id,
                        "config_label": cell.config_label,
                        "phase": cell.phase,
                        "level": int(cell.level),
                        "bias": bool(cell.bias),
                        "imbalance": bool(cell.imbalance),
                        "problem": problem,
                        "algorithm": algorithm,
                        "algorithm_label": ALGORITHM_LABELS[algorithm],
                        "runs": len(runs),
                        "hv_mean": hv_mean,
                        "hv_std": hv_std,
                        "hv_median": _median_or_zero(hv_values),
                        "igd_plus_mean": igd_mean,
                        "igd_plus_std": igd_std,
                        "igd_plus_median": _median_or_zero(igd_values),
                        "runtime_seconds_mean": runtime_mean,
                        "runtime_seconds_std": runtime_std,
                        "runtime_seconds_median": _median_or_zero(runtime_values),
                    }
                )

            for lhs_algorithm, rhs_algorithm in comparisons:
                lhs_runs = {run.seed: run for run in grouped.get((cell.config_id, problem, lhs_algorithm), [])}
                rhs_runs = {run.seed: run for run in grouped.get((cell.config_id, problem, rhs_algorithm), [])}
                common_seeds = sorted(set(lhs_runs) & set(rhs_runs))
                if not common_seeds:
                    continue
                lhs_hv = [lhs_runs[seed].hypervolume for seed in common_seeds]
                rhs_hv = [rhs_runs[seed].hypervolume for seed in common_seeds]
                lhs_igd = [lhs_runs[seed].igd_plus for seed in common_seeds]
                rhs_igd = [rhs_runs[seed].igd_plus for seed in common_seeds]
                comparison_rows.append(
                    _build_metric_row(
                        cell=cell,
                        problem=problem,
                        lhs_algorithm=lhs_algorithm,
                        rhs_algorithm=rhs_algorithm,
                        lhs_label=ALGORITHM_LABELS[lhs_algorithm],
                        rhs_label=ALGORITHM_LABELS[rhs_algorithm],
                        metric="hypervolume",
                        higher_is_better=True,
                        lhs_values=lhs_hv,
                        rhs_values=rhs_hv,
                        tie_tol=tie_tol,
                    )
                )
                comparison_rows.append(
                    _build_metric_row(
                        cell=cell,
                        problem=problem,
                        lhs_algorithm=lhs_algorithm,
                        rhs_algorithm=rhs_algorithm,
                        lhs_label=ALGORITHM_LABELS[lhs_algorithm],
                        rhs_label=ALGORITHM_LABELS[rhs_algorithm],
                        metric="igd_plus",
                        higher_is_better=False,
                        lhs_values=lhs_igd,
                        rhs_values=rhs_igd,
                        tie_tol=tie_tol,
                    )
                )

    for cell in cells:
        for metric in ("hypervolume", "igd_plus"):
            metric_rows = [row for row in comparison_rows if row["config_id"] == cell.config_id and row["metric"] == metric]
            adjusted = holm_adjust([row["wilcoxon_p_value_raw"] for row in metric_rows])
            for row, adj_p in zip(metric_rows, adjusted, strict=True):
                row["wilcoxon_p_value_holm"] = adj_p
                row["wilcoxon_significant_alpha"] = bool(adj_p is not None and adj_p < alpha)

    summary_lookup = _summary_lookup(summary_rows)
    global_summary_rows: list[dict[str, Any]] = []
    for cell in cells:
        for algorithm in algorithms:
            rows = [summary_lookup[(cell.config_id, problem, algorithm)] for problem in problems if (cell.config_id, problem, algorithm) in summary_lookup]
            if not rows:
                continue
            hv_problem_medians = [float(row["hv_median"]) for row in rows]
            igd_problem_medians = [float(row["igd_plus_median"]) for row in rows]
            runtime_problem_medians = [float(row["runtime_seconds_median"]) for row in rows]
            global_summary_rows.append(
                {
                    "config_id": cell.config_id,
                    "config_label": cell.config_label,
                    "phase": cell.phase,
                    "level": int(cell.level),
                    "bias": bool(cell.bias),
                    "imbalance": bool(cell.imbalance),
                    "algorithm": algorithm,
                    "algorithm_label": ALGORITHM_LABELS[algorithm],
                    "problem_count": len(rows),
                    "hv_median_of_medians": _median_or_zero(hv_problem_medians),
                    "igd_plus_median_of_medians": _median_or_zero(igd_problem_medians),
                    "runtime_seconds_median_of_medians": _median_or_zero(runtime_problem_medians),
                }
            )

    global_lookup = _global_summary_lookup(global_summary_rows)
    for cell in cells:
        baseline = global_lookup.get((cell.config_id, "nsgaii"))
        if baseline is None:
            continue
        hv_baseline = float(baseline["hv_median_of_medians"])
        igd_baseline = float(baseline["igd_plus_median_of_medians"])
        runtime_baseline = float(baseline["runtime_seconds_median_of_medians"])
        for algorithm in algorithms:
            row = global_lookup.get((cell.config_id, algorithm))
            if row is None:
                continue
            hv_value = float(row["hv_median_of_medians"])
            igd_value = float(row["igd_plus_median_of_medians"])
            runtime_value = float(row["runtime_seconds_median_of_medians"])
            row["hv_delta_vs_nsgaii"] = hv_value - hv_baseline
            row["igd_plus_delta_vs_nsgaii"] = igd_value - igd_baseline
            row["igd_plus_improvement_vs_nsgaii"] = igd_baseline - igd_value
            row["runtime_seconds_delta_vs_nsgaii"] = runtime_value - runtime_baseline
            row["runtime_ratio_vs_nsgaii"] = None if runtime_baseline <= 0.0 else runtime_value / runtime_baseline

    global_comparison_rows: list[dict[str, Any]] = []
    for cell in cells:
        for lhs_algorithm, rhs_algorithm in comparisons:
            for metric in ("hypervolume", "igd_plus"):
                rows = [
                    row
                    for row in comparison_rows
                    if row["config_id"] == cell.config_id
                    and row["lhs_algorithm"] == lhs_algorithm
                    and row["rhs_algorithm"] == rhs_algorithm
                    and row["metric"] == metric
                ]
                if not rows:
                    continue
                better = 0
                worse = 0
                ties = 0
                for row in rows:
                    delta = float(row["delta_median"])
                    if abs(delta) <= tie_tol:
                        ties += 1
                    elif (metric == "hypervolume" and delta > 0.0) or (metric == "igd_plus" and delta < 0.0):
                        better += 1
                    else:
                        worse += 1
                global_comparison_rows.append(
                    {
                        "config_id": cell.config_id,
                        "config_label": cell.config_label,
                        "phase": cell.phase,
                        "level": int(cell.level),
                        "bias": bool(cell.bias),
                        "imbalance": bool(cell.imbalance),
                        "lhs_algorithm": lhs_algorithm,
                        "rhs_algorithm": rhs_algorithm,
                        "lhs_label": ALGORITHM_LABELS[lhs_algorithm],
                        "rhs_label": ALGORITHM_LABELS[rhs_algorithm],
                        "metric": metric,
                        "problem_count": len(rows),
                        "problems_better": int(better),
                        "problems_tied": int(ties),
                        "problems_worse": int(worse),
                        "median_delta_across_problems": _median_or_zero([float(row["delta_median"]) for row in rows]),
                        "lhs_seed_wins_total": int(sum(int(row["lhs_seed_wins"]) for row in rows)),
                        "seed_ties_total": int(sum(int(row["seed_ties"]) for row in rows)),
                        "rhs_seed_wins_total": int(sum(int(row["rhs_seed_wins"]) for row in rows)),
                        "significant_problem_count": int(sum(1 for row in rows if bool(row["wilcoxon_significant_alpha"]))),
                    }
                )

    degradation_rows: list[dict[str, Any]] = []
    base_config_id = "lvl1_plain"
    for algorithm in algorithms:
        base_row = global_lookup.get((base_config_id, algorithm))
        base_nsga = global_lookup.get((base_config_id, "nsgaii"))
        if base_row is None or base_nsga is None:
            continue
        base_hv_adv = float(base_row["hv_median_of_medians"]) - float(base_nsga["hv_median_of_medians"])
        base_igd_adv = float(base_nsga["igd_plus_median_of_medians"]) - float(base_row["igd_plus_median_of_medians"])
        for cell in cells:
            row = global_lookup.get((cell.config_id, algorithm))
            nsga_row = global_lookup.get((cell.config_id, "nsgaii"))
            if row is None or nsga_row is None:
                continue
            current_hv_adv = float(row["hv_median_of_medians"]) - float(nsga_row["hv_median_of_medians"])
            current_igd_adv = float(nsga_row["igd_plus_median_of_medians"]) - float(row["igd_plus_median_of_medians"])
            degradation_rows.append(
                {
                    "config_id": cell.config_id,
                    "config_label": cell.config_label,
                    "phase": cell.phase,
                    "level": int(cell.level),
                    "bias": bool(cell.bias),
                    "imbalance": bool(cell.imbalance),
                    "algorithm": algorithm,
                    "algorithm_label": ALGORITHM_LABELS[algorithm],
                    "hv_delta_vs_lvl1_plain": float(row["hv_median_of_medians"]) - float(base_row["hv_median_of_medians"]),
                    "igd_plus_delta_vs_lvl1_plain": float(row["igd_plus_median_of_medians"]) - float(base_row["igd_plus_median_of_medians"]),
                    "runtime_seconds_delta_vs_lvl1_plain": float(row["runtime_seconds_median_of_medians"]) - float(base_row["runtime_seconds_median_of_medians"]),
                    "hv_advantage_vs_nsgaii": current_hv_adv,
                    "hv_advantage_retention_vs_lvl1_plain": None if abs(base_hv_adv) <= 1e-15 else current_hv_adv / base_hv_adv,
                    "igd_plus_advantage_vs_nsgaii": current_igd_adv,
                    "igd_plus_advantage_retention_vs_lvl1_plain": None if abs(base_igd_adv) <= 1e-15 else current_igd_adv / base_igd_adv,
                }
            )

    structure_effect_rows: list[dict[str, Any]] = []
    structure_targets = ("lvl6_bias", "lvl6_imbalance", "lvl6_bias_imbalance")
    for algorithm in algorithms:
        plain_row = global_lookup.get(("lvl6_plain", algorithm))
        plain_nsga = global_lookup.get(("lvl6_plain", "nsgaii"))
        if plain_row is None or plain_nsga is None:
            continue
        plain_hv_adv = float(plain_row["hv_median_of_medians"]) - float(plain_nsga["hv_median_of_medians"])
        plain_igd_adv = float(plain_nsga["igd_plus_median_of_medians"]) - float(plain_row["igd_plus_median_of_medians"])
        for config_id in structure_targets:
            row = global_lookup.get((config_id, algorithm))
            nsga_row = global_lookup.get((config_id, "nsgaii"))
            if row is None or nsga_row is None:
                continue
            structure_effect_rows.append(
                {
                    "config_id": config_id,
                    "config_label": row["config_label"],
                    "phase": row["phase"],
                    "level": int(row["level"]),
                    "bias": bool(row["bias"]),
                    "imbalance": bool(row["imbalance"]),
                    "algorithm": algorithm,
                    "algorithm_label": ALGORITHM_LABELS[algorithm],
                    "hv_delta_vs_lvl6_plain": float(row["hv_median_of_medians"]) - float(plain_row["hv_median_of_medians"]),
                    "igd_plus_delta_vs_lvl6_plain": float(row["igd_plus_median_of_medians"]) - float(plain_row["igd_plus_median_of_medians"]),
                    "runtime_seconds_delta_vs_lvl6_plain": float(row["runtime_seconds_median_of_medians"]) - float(plain_row["runtime_seconds_median_of_medians"]),
                    "hv_advantage_change_vs_nsgaii": (
                        float(row["hv_median_of_medians"]) - float(nsga_row["hv_median_of_medians"]) - plain_hv_adv
                    ),
                    "igd_plus_advantage_change_vs_nsgaii": (
                        float(nsga_row["igd_plus_median_of_medians"]) - float(row["igd_plus_median_of_medians"]) - plain_igd_adv
                    ),
                }
            )

    return (
        summary_rows,
        comparison_rows,
        global_summary_rows,
        global_comparison_rows,
        degradation_rows,
        structure_effect_rows,
    )


def generate_markdown_report(
    *,
    output_path: Path,
    summary_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    global_summary_rows: list[dict[str, Any]],
    global_comparison_rows: list[dict[str, Any]],
    degradation_rows: list[dict[str, Any]],
    structure_effect_rows: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> None:
    algorithms = list(run_config["algorithms"])
    algorithm_labels = {
        algorithm: run_config.get("algorithm_labels", {}).get(algorithm, ALGORITHM_LABELS.get(algorithm, algorithm))
        for algorithm in algorithms
    }
    cells = [DifficultyCell(**item) for item in run_config["difficulty_cells"]]

    hv_global_rows: list[list[str]] = []
    igd_global_rows: list[list[str]] = []
    runtime_global_rows: list[list[str]] = []
    versus_nsga_rows: list[list[str]] = []
    for cell in cells:
        hv_row = [cell.config_label]
        igd_row = [cell.config_label]
        rt_row = [cell.config_label]
        for algorithm in algorithms:
            row = next(item for item in global_summary_rows if item["config_id"] == cell.config_id and item["algorithm"] == algorithm)
            hv_row.append(_format_number(row["hv_median_of_medians"]))
            igd_row.append(_format_number(row["igd_plus_median_of_medians"]))
            rt_row.append(_format_number(row["runtime_seconds_median_of_medians"], digits=3))
            if algorithm != "nsgaii":
                versus_nsga_rows.append(
                    [
                        cell.config_label,
                        algorithm_labels[algorithm],
                        _format_number(row["hv_delta_vs_nsgaii"]),
                        _format_number(row["igd_plus_improvement_vs_nsgaii"]),
                        _format_number(row["runtime_ratio_vs_nsgaii"], digits=3),
                    ]
                )
        hv_global_rows.append(hv_row)
        igd_global_rows.append(igd_row)
        runtime_global_rows.append(rt_row)

    degradation_table_rows: list[list[str]] = []
    for row in degradation_rows:
        if row["algorithm"] == "nsgaii":
            continue
        degradation_table_rows.append(
            [
                str(row["config_label"]),
                str(row["algorithm_label"]),
                _format_number(row["hv_advantage_vs_nsgaii"]),
                _format_number(row["hv_advantage_retention_vs_lvl1_plain"], digits=3),
                _format_number(row["igd_plus_advantage_vs_nsgaii"]),
                _format_number(row["igd_plus_advantage_retention_vs_lvl1_plain"], digits=3),
            ]
        )

    structure_table_rows: list[list[str]] = []
    for row in structure_effect_rows:
        if row["algorithm"] == "nsgaii":
            continue
        structure_table_rows.append(
            [
                str(row["config_label"]),
                str(row["algorithm_label"]),
                _format_number(row["hv_delta_vs_lvl6_plain"]),
                _format_number(row["igd_plus_delta_vs_lvl6_plain"]),
                _format_number(row["hv_advantage_change_vs_nsgaii"]),
                _format_number(row["igd_plus_advantage_change_vs_nsgaii"]),
            ]
        )

    best_by_cell_rows: list[list[str]] = []
    for cell in cells:
        cell_rows = [row for row in global_summary_rows if row["config_id"] == cell.config_id]
        best_hv = max(cell_rows, key=lambda row: float(row["hv_median_of_medians"]))
        best_igd = min(cell_rows, key=lambda row: float(row["igd_plus_median_of_medians"]))
        best_by_cell_rows.append([cell.config_label, str(best_hv["algorithm_label"]), str(best_igd["algorithm_label"])])

    report_title = str(run_config.get("report_title", "3-objective ZCAT difficulty robustness campaign"))
    report_note = str(
        run_config.get(
            "report_note",
            "This campaign keeps the NSGA-II host fixed and varies only split-front survival behavior across algorithms.",
        )
    )

    content = [
        f"# {report_title}",
        "",
        report_note,
        "",
        "## Settings",
        "",
        f"- Problems: {', '.join(run_config['problems'])}",
        f"- Algorithms: {', '.join(algorithm_labels[algorithm] for algorithm in algorithms)}",
        f"- Seeds: {run_config['seeds']}",
        f"- Engine: {run_config['engine']}",
        f"- Population size: {run_config['pop_size']}",
        f"- Max evaluations: {run_config['max_evaluations']}",
        f"- Decision variables: {run_config['n_var']}",
        f"- Objectives: {run_config['n_obj']}",
        f"- Worker count: {run_config['workers']}",
        f"- Total expected runs: {run_config['expected_total_runs']}",
        "",
        "## Difficulty Cells",
        "",
        _markdown_table(
            ["Config", "Phase", "Level", "Bias", "Imbalance"],
            [[cell.config_label, cell.phase, str(cell.level), str(cell.bias), str(cell.imbalance)] for cell in cells],
        ),
        "",
        "## Global Median-of-Medians",
        "",
        "### Hypervolume",
        "",
        _markdown_table(["Config"] + [algorithm_labels[algorithm] for algorithm in algorithms], hv_global_rows),
        "",
        "### IGD+",
        "",
        _markdown_table(["Config"] + [algorithm_labels[algorithm] for algorithm in algorithms], igd_global_rows),
        "",
        "### Runtime (seconds)",
        "",
        _markdown_table(["Config"] + [algorithm_labels[algorithm] for algorithm in algorithms], runtime_global_rows),
        "",
        "## Global Pairwise View Against nsgaii",
        "",
        _markdown_table(["Config", "Algorithm", "HV delta", "IGD+ improvement", "Runtime ratio"], versus_nsga_rows),
        "",
        "## Advantage Retention Relative to L1",
        "",
        _markdown_table(
            ["Config", "Algorithm", "HV advantage", "HV retention", "IGD+ advantage", "IGD+ retention"],
            degradation_table_rows,
        ),
        "",
        "## Additional Effect of Bias / Imbalance at Level 6",
        "",
        _markdown_table(
            ["Config", "Algorithm", "HV vs L6", "IGD+ vs L6", "HV advantage shift", "IGD+ advantage shift"],
            structure_table_rows,
        ),
        "",
        "## Best Method by Difficulty Cell",
        "",
        _markdown_table(["Config", "Best HV", "Best IGD+"], best_by_cell_rows),
        "",
        "## Notes",
        "",
        "- `summary.csv` contains per-problem per-algorithm medians and dispersion by difficulty cell.",
        "- `comparison.csv` contains paired Wilcoxon signed-rank tests against `nsgaii` for each problem and difficulty cell.",
        "- `global_summary.csv` collects median-of-medians and runtime trade-offs by difficulty cell.",
        "- `difficulty_degradation.csv` and `difficulty_structure_effects.csv` quantify how quality changes as difficulty increases.",
        "",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def write_analysis_artifacts(
    *,
    output_dir: Path,
    records: list[RunRecord],
    run_config: dict[str, Any],
    alpha: float,
    tie_tol: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons = _coerce_comparisons(run_config.get("comparisons", DEFAULT_COMPARISONS))
    algorithm_order = list(run_config.get("algorithms", DEFAULT_ALGORITHMS))
    (
        summary_rows,
        comparison_rows,
        global_summary_rows,
        global_comparison_rows,
        degradation_rows,
        structure_effect_rows,
    ) = summarize_records(
        records,
        comparisons=comparisons,
        algorithm_order=algorithm_order,
        alpha=alpha,
        tie_tol=tie_tol,
    )
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "comparison.csv", comparison_rows)
    write_csv(output_dir / "global_summary.csv", global_summary_rows)
    write_csv(output_dir / "global_comparison.csv", global_comparison_rows)
    write_csv(output_dir / "difficulty_degradation.csv", degradation_rows)
    write_csv(output_dir / "difficulty_structure_effects.csv", structure_effect_rows)
    generate_markdown_report(
        output_path=output_dir / "report.md",
        summary_rows=summary_rows,
        comparison_rows=comparison_rows,
        global_summary_rows=global_summary_rows,
        global_comparison_rows=global_comparison_rows,
        degradation_rows=degradation_rows,
        structure_effect_rows=structure_effect_rows,
        run_config=run_config,
    )
    return (
        summary_rows,
        comparison_rows,
        global_summary_rows,
        global_comparison_rows,
        degradation_rows,
        structure_effect_rows,
    )


__all__ = [
    "ALGORITHM_LABELS",
    "DEFAULT_ALGORITHMS",
    "DEFAULT_COMPARISONS",
    "DEFAULT_DIFFICULTY_CELLS",
    "DEFAULT_ENGINE",
    "DEFAULT_EVALS",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_N_OBJ",
    "DEFAULT_N_VAR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_POP_SIZE",
    "DEFAULT_PROBLEMS",
    "DEFAULT_SEEDS",
    "DifficultyCell",
    "ROOT_DIR",
    "RunRecord",
    "generate_markdown_report",
    "instantiate_zcat_problem",
    "load_raw_records",
    "resolve_worker_count",
    "run_campaign",
    "run_once",
    "select_difficulty_cells",
    "summarize_records",
    "write_analysis_artifacts",
    "write_csv",
    "write_run_config",
]
