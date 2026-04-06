"""
Shared helpers for the full 2-objective ZCAT GCES ablation.

The default settings intentionally mirror the existing 2-objective GCES E2
campaign in ``paper/34_run_gces_e1.py``:
- problems: zcat1..zcat20
- objectives: 2
- n_var: 30
- pop_size: 100
- max_evaluations: 25_000
- seeds: 0..20
- engine: numpy
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

try:
    from .benchmark_utils import compute_hv, compute_igd_plus
except ImportError:
    from benchmark_utils import compute_hv, compute_igd_plus


DEFAULT_PROBLEMS = [f"zcat{i}" for i in range(1, 21)]
ALGORITHM_LABELS = {
    "nsgaii": "nsgaii",
    "nsga2_farthest": "nsga2-farthest",
    "nsga2_gapfill": "nsga2-gapfill",
    "nsga2_curvgap": "nsga2-curvgap",
    "nsga2_hvfarthest": "nsga2-hvfarthest",
    "nsga2_refcover_farthest": "nsga2-refcover-farthest",
    "nsga2_hvref_farthest": "nsga2-hvref-farthest",
    "nsga2_sector_farthest": "nsga2-sector-farthest",
    "gces_nocomp": "gces-noComp",
    "gces_nogeo": "gces-noGeo",
    "gces": "gces",
}
DEFAULT_ALGORITHMS = ["nsgaii", "nsga2_farthest", "gces_nocomp", "gces_nogeo", "gces"]
DEFAULT_COMPARISONS = [
    ("gces", "nsgaii"),
    ("gces_nocomp", "nsgaii"),
    ("gces_nogeo", "nsgaii"),
    ("gces", "gces_nocomp"),
    ("gces", "gces_nogeo"),
]
DEFAULT_POP_SIZE = 100
DEFAULT_EVALS = 25_000
DEFAULT_SEEDS = 21
DEFAULT_N_VAR = 30
DEFAULT_N_OBJ = 2
DEFAULT_ENGINE = "numpy"
DEFAULT_MAX_WORKERS = 16
DEFAULT_OUTPUT_DIR = ROOT_DIR / "experiments" / "gces_ablation_zcat_2obj_full"


@dataclass(frozen=True)
class RunRecord:
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
    sep = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _coerce_comparisons(comparisons: Sequence[tuple[str, str]] | Sequence[Sequence[str]]) -> list[tuple[str, str]]:
    coerced: list[tuple[str, str]] = []
    for pair in comparisons:
        if len(pair) != 2:
            raise ValueError(f"Invalid comparison pair: {pair!r}")
        coerced.append((str(pair[0]), str(pair[1])))
    return coerced


def build_config(pop_size: int, n_var: int) -> NSGAIIConfig:
    return (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("polynomial", prob=1.0 / n_var, eta=20.0)
        .selection("tournament")
        .build()
    )


def metric_problem_name(problem: str, n_obj: int) -> str:
    key = str(problem).strip().lower()
    if key.startswith("zcat") and "." not in key and int(n_obj) > 2:
        return f"{key}.{int(n_obj)}d"
    return key


def resolve_worker_count(workers: int | None) -> int:
    available = max(1, int(os.cpu_count() or 1))
    if workers is None:
        return max(1, min(DEFAULT_MAX_WORKERS, available))
    return max(1, min(int(workers), available))


def run_once(
    *,
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
    start = time.perf_counter()
    result = optimize(
        problem,
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
        problem=problem,
        algorithm=algorithm,
        algorithm_label=ALGORITHM_LABELS[algorithm],
        seed=seed,
        engine=engine,
        pop_size=pop_size,
        max_evaluations=max_evaluations,
        n_var=n_var,
        n_obj=n_obj,
        runtime_seconds=float(elapsed),
        n_solutions=int(F.shape[0]),
        hypervolume=float(compute_hv(F, metric_problem)),
        igd_plus=float(compute_igd_plus(F, metric_problem)),
    )


def _task_order_key(task: dict[str, Any]) -> tuple[int, int, int]:
    return int(task["problem_index"]), int(task["algorithm_index"]), int(task["seed"])


def _run_task(task: dict[str, Any]) -> tuple[tuple[int, int, int], RunRecord]:
    record = run_once(
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
    return _task_order_key(task), record


def _format_progress_line(record: RunRecord, completed: int, total: int) -> str:
    return (
        f"[{completed:>4}/{total}] "
        f"{record.problem:<7} | "
        f"{record.algorithm_label:>24} seed={record.seed:>2} | "
        f"HV={record.hypervolume:.4f} | "
        f"IGD+={record.igd_plus:.4f} | "
        f"time={record.runtime_seconds:.2f}s"
    )


def run_campaign(
    *,
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
    for problem_index, problem in enumerate(problems):
        for algorithm_index, algorithm in enumerate(algorithms):
            for seed in seeds:
                tasks.append(
                    {
                        "problem": str(problem),
                        "algorithm": str(algorithm),
                        "seed": int(seed),
                        "engine": str(engine),
                        "pop_size": int(pop_size),
                        "max_evaluations": int(max_evaluations),
                        "n_var": int(n_var),
                        "n_obj": int(n_obj),
                        "problem_index": int(problem_index),
                        "algorithm_index": int(algorithm_index),
                    }
                )

    indexed_records: list[tuple[tuple[int, int, int], RunRecord]] = []
    total = len(tasks)
    completed = 0

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
                        "Campaign run failed for "
                        f"problem={task['problem']!r}, algorithm={task['algorithm']!r}, seed={task['seed']!r}."
                    ) from exc
                indexed_records.append((order_key, record))
                completed += 1
                if show_progress:
                    print(_format_progress_line(record, completed, total), flush=True)

    indexed_records.sort(key=lambda item: item[0])
    return [record for _, record in indexed_records]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_run_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_run_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_raw_records(path: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            records.append(
                RunRecord(
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


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    p = np.asarray([np.nan if value is None else float(value) for value in p_values], dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite_idx = np.where(np.isfinite(p))[0]
    if finite_idx.size == 0:
        return [None for _ in p_values]

    m = int(finite_idx.size)
    order = finite_idx[np.argsort(p[finite_idx])]
    prev = 0.0
    for k, idx in enumerate(order):
        adj = min(1.0, float(p[idx]) * float(m - k))
        adj = max(prev, adj)
        prev = adj
        out[idx] = adj
    return [None if not np.isfinite(value) else float(value) for value in out.tolist()]


def paired_wilcoxon_pvalue(lhs_values: list[float], rhs_values: list[float]) -> float | None:
    if not lhs_values or len(lhs_values) != len(rhs_values):
        return None

    lhs = np.asarray(lhs_values, dtype=float)
    rhs = np.asarray(rhs_values, dtype=float)
    if lhs.shape != rhs.shape:
        return None
    if np.allclose(lhs, rhs, atol=1e-15, rtol=0.0):
        return 1.0

    try:
        from scipy import stats as spstats  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - defensive path for non-paper envs
        raise RuntimeError("SciPy is required to compute Wilcoxon p-values for the GCES ablation report.") from exc

    try:
        _stat, p_value = spstats.wilcoxon(lhs, rhs, zero_method="pratt", alternative="two-sided", method="auto")
    except TypeError:  # pragma: no cover - older SciPy fallback
        _stat, p_value = spstats.wilcoxon(lhs, rhs, zero_method="pratt", alternative="two-sided")
    except ValueError:
        return None
    return float(p_value)


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


def _both_metric_wins(
    lhs_hv: list[float],
    rhs_hv: list[float],
    lhs_igd: list[float],
    rhs_igd: list[float],
    *,
    tie_tol: float,
) -> tuple[int, int, int]:
    wins = 0
    ties = 0
    losses = 0
    for hv_l, hv_r, igd_l, igd_r in zip(lhs_hv, rhs_hv, lhs_igd, rhs_igd, strict=True):
        hv_diff = float(hv_l - hv_r)
        igd_diff = float(igd_l - igd_r)
        if abs(hv_diff) <= tie_tol and abs(igd_diff) <= tie_tol:
            ties += 1
        elif hv_diff > tie_tol and igd_diff < -tie_tol:
            wins += 1
        elif hv_diff < -tie_tol and igd_diff > tie_tol:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses


def _build_metric_row(
    *,
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
    both_wins: tuple[int, int, int],
) -> dict[str, Any]:
    wins, ties, losses = _win_tie_loss(lhs_values, rhs_values, higher_is_better=higher_is_better, tie_tol=tie_tol)
    return {
        "problem": problem,
        "metric": metric,
        "higher_is_better": higher_is_better,
        "lhs_algorithm": lhs_algorithm,
        "lhs_label": lhs_label,
        "rhs_algorithm": rhs_algorithm,
        "rhs_label": rhs_label,
        "n_common_seeds": len(lhs_values),
        "lhs_mean": float(mean(lhs_values)),
        "lhs_median": _median_or_zero(lhs_values),
        "rhs_mean": float(mean(rhs_values)),
        "rhs_median": _median_or_zero(rhs_values),
        "delta_mean": float(mean(lhs_values) - mean(rhs_values)),
        "delta_median": _median_or_zero(lhs_values) - _median_or_zero(rhs_values),
        "lhs_seed_wins": int(wins),
        "seed_ties": int(ties),
        "rhs_seed_wins": int(losses),
        "lhs_both_metric_seed_wins": int(both_wins[0]),
        "both_metric_seed_ties_or_mixed": int(both_wins[1]),
        "rhs_both_metric_seed_wins": int(both_wins[2]),
        "wilcoxon_p_value_raw": paired_wilcoxon_pvalue(lhs_values, rhs_values),
        "wilcoxon_p_value_holm": None,
        "wilcoxon_significant_alpha": False,
    }


def summarize_records(
    records: list[RunRecord],
    *,
    comparisons: Sequence[tuple[str, str]] = DEFAULT_COMPARISONS,
    algorithm_order: Sequence[str] | None = None,
    alpha: float = 0.05,
    tie_tol: float = 1e-12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.problem, record.algorithm)].append(record)

    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    problems = sorted({record.problem for record in records})
    ordered_algorithms = list(DEFAULT_ALGORITHMS if algorithm_order is None else algorithm_order)
    algorithms = [algorithm for algorithm in ordered_algorithms if any((problem, algorithm) in grouped for problem in problems)]

    for problem in problems:
        for algorithm in algorithms:
            runs = grouped.get((problem, algorithm), [])
            if not runs:
                continue
            hv_values = [run.hypervolume for run in runs]
            igd_values = [run.igd_plus for run in runs]
            rt_values = [run.runtime_seconds for run in runs]
            hv_mean, hv_std = _float_or_zero(hv_values)
            igd_mean, igd_std = _float_or_zero(igd_values)
            rt_mean, rt_std = _float_or_zero(rt_values)
            summary_rows.append(
                {
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
                    "runtime_seconds_mean": rt_mean,
                    "runtime_seconds_std": rt_std,
                    "runtime_seconds_median": _median_or_zero(rt_values),
                }
            )

        for lhs_algorithm, rhs_algorithm in comparisons:
            lhs_runs = {run.seed: run for run in grouped.get((problem, lhs_algorithm), [])}
            rhs_runs = {run.seed: run for run in grouped.get((problem, rhs_algorithm), [])}
            common_seeds = sorted(set(lhs_runs) & set(rhs_runs))
            if not common_seeds:
                continue

            lhs_hv = [lhs_runs[seed].hypervolume for seed in common_seeds]
            rhs_hv = [rhs_runs[seed].hypervolume for seed in common_seeds]
            lhs_igd = [lhs_runs[seed].igd_plus for seed in common_seeds]
            rhs_igd = [rhs_runs[seed].igd_plus for seed in common_seeds]
            both_wins = _both_metric_wins(lhs_hv, rhs_hv, lhs_igd, rhs_igd, tie_tol=tie_tol)

            comparison_rows.append(
                _build_metric_row(
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
                    both_wins=both_wins,
                )
            )
            comparison_rows.append(
                _build_metric_row(
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
                    both_wins=both_wins,
                )
            )

    for metric in ("hypervolume", "igd_plus"):
        metric_rows = [row for row in comparison_rows if row["metric"] == metric]
        adjusted = holm_adjust([row["wilcoxon_p_value_raw"] for row in metric_rows])
        for row, adj_p in zip(metric_rows, adjusted, strict=True):
            row["wilcoxon_p_value_holm"] = adj_p
            row["wilcoxon_significant_alpha"] = bool(adj_p is not None and adj_p < alpha)

    return summary_rows, comparison_rows


def _summary_lookup(summary_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["problem"], row["algorithm"]): row for row in summary_rows}


def generate_markdown_report(
    *,
    output_path: Path,
    summary_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> None:
    problems = list(run_config["problems"])
    algorithms = list(run_config["algorithms"])
    algorithm_labels = {
        algorithm: run_config.get("algorithm_labels", {}).get(algorithm, ALGORITHM_LABELS.get(algorithm, algorithm))
        for algorithm in algorithms
    }
    comparisons = _coerce_comparisons(run_config.get("comparisons", DEFAULT_COMPARISONS))
    summary_map = _summary_lookup(summary_rows)
    report_title = str(run_config.get("report_title", "GCES 2-objective ZCAT survival-only ablation"))
    report_note = str(
        run_config.get(
            "report_note",
            "This report describes a survival-only ablation. All algorithms in this campaign reuse the NSGA-II host, mating, variation, and non-dominated sorting. Only split-front environmental selection differs across the compared variants.",
        )
    )

    hv_table_rows: list[list[str]] = []
    igd_table_rows: list[list[str]] = []
    runtime_table_rows: list[list[str]] = []
    for problem in problems:
        hv_table_rows.append(
            [problem]
            + [_format_number(summary_map[(problem, algorithm)]["hv_median"]) for algorithm in algorithms]
        )
        igd_table_rows.append(
            [problem]
            + [_format_number(summary_map[(problem, algorithm)]["igd_plus_median"]) for algorithm in algorithms]
        )
        runtime_table_rows.append(
            [problem]
            + [_format_number(summary_map[(problem, algorithm)]["runtime_seconds_median"], digits=3) for algorithm in algorithms]
        )

    win_rows: list[list[str]] = []
    for problem in problems:
        for lhs_algorithm, rhs_algorithm in comparisons:
            hv_row = next(
                row
                for row in comparison_rows
                if row["problem"] == problem and row["metric"] == "hypervolume" and row["lhs_algorithm"] == lhs_algorithm and row["rhs_algorithm"] == rhs_algorithm
            )
            igd_row = next(
                row
                for row in comparison_rows
                if row["problem"] == problem and row["metric"] == "igd_plus" and row["lhs_algorithm"] == lhs_algorithm and row["rhs_algorithm"] == rhs_algorithm
            )
            win_rows.append(
                [
                    problem,
                    f"{hv_row['lhs_label']} vs {hv_row['rhs_label']}",
                    f"{hv_row['lhs_seed_wins']}/{hv_row['seed_ties']}/{hv_row['rhs_seed_wins']}",
                    f"{igd_row['lhs_seed_wins']}/{igd_row['seed_ties']}/{igd_row['rhs_seed_wins']}",
                    f"{hv_row['lhs_both_metric_seed_wins']}/{hv_row['both_metric_seed_ties_or_mixed']}/{hv_row['rhs_both_metric_seed_wins']}",
                ]
            )

    def metric_rows(metric: str) -> list[list[str]]:
        rows: list[list[str]] = []
        for row in comparison_rows:
            if row["metric"] != metric:
                continue
            rows.append(
                [
                    row["problem"],
                    f"{row['lhs_label']} vs {row['rhs_label']}",
                    _format_number(row["delta_median"]),
                    f"{row['lhs_seed_wins']}/{row['seed_ties']}/{row['rhs_seed_wins']}",
                    _format_number(row["wilcoxon_p_value_raw"]),
                    _format_number(row["wilcoxon_p_value_holm"]),
                    "yes" if row["wilcoxon_significant_alpha"] else "no",
                ]
            )
        return rows

    content = [
        f"# {report_title}",
        "",
        report_note,
        "",
        "## Settings",
        "",
        f"- Problems: {', '.join(problems)}",
        f"- Algorithms: {', '.join(algorithm_labels[algorithm] for algorithm in algorithms)}",
        f"- Seeds: {run_config['seeds']}",
        f"- Engine: {run_config['engine']}",
        f"- Population size: {run_config['pop_size']}",
        f"- Max evaluations: {run_config['max_evaluations']}",
        f"- Decision variables: {run_config['n_var']}",
        f"- Objectives: {run_config['n_obj']}",
        f"- Tie tolerance: {run_config['tie_tol']}",
        f"- Wilcoxon alpha: {run_config['alpha']}",
        f"- Pairwise comparisons: {', '.join(f'{algorithm_labels[lhs]} vs {algorithm_labels[rhs]}' for lhs, rhs in comparisons)}",
        "",
        "## Median Hypervolume by Problem and Algorithm",
        "",
        _markdown_table(["Problem"] + [algorithm_labels[algorithm] for algorithm in algorithms], hv_table_rows),
        "",
        "## Median IGD+ by Problem and Algorithm",
        "",
        _markdown_table(["Problem"] + [algorithm_labels[algorithm] for algorithm in algorithms], igd_table_rows),
        "",
        "## Median Runtime (seconds) by Problem and Algorithm",
        "",
        _markdown_table(["Problem"] + [algorithm_labels[algorithm] for algorithm in algorithms], runtime_table_rows),
        "",
        "## Seed Win Counts",
        "",
        _markdown_table(
            ["Problem", "Comparison", "HV W/T/L", "IGD+ W/T/L", "Both-metric W/T/L"],
            win_rows,
        ),
        "",
        "## Paired Wilcoxon Signed-Rank Tests with Holm Correction",
        "",
        "Holm correction is applied within each metric family across all problem-level pairwise tests in that metric.",
        "",
        "### Hypervolume",
        "",
        _markdown_table(
            ["Problem", "Comparison", "Median Delta", "W/T/L", "p_raw", "p_holm", "significant"],
            metric_rows("hypervolume"),
        ),
        "",
        "### IGD+",
        "",
        _markdown_table(
            ["Problem", "Comparison", "Median Delta", "W/T/L", "p_raw", "p_holm", "significant"],
            metric_rows("igd_plus"),
        ),
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons = _coerce_comparisons(run_config.get("comparisons", DEFAULT_COMPARISONS))
    algorithm_order = list(run_config.get("algorithms", DEFAULT_ALGORITHMS))
    summary_rows, comparison_rows = summarize_records(
        records,
        comparisons=comparisons,
        algorithm_order=algorithm_order,
        alpha=alpha,
        tie_tol=tie_tol,
    )
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "comparison.csv", comparison_rows)
    generate_markdown_report(
        output_path=output_dir / "report.md",
        summary_rows=summary_rows,
        comparison_rows=comparison_rows,
        run_config=run_config,
    )
    return summary_rows, comparison_rows


__all__ = [
    "ALGORITHM_LABELS",
    "DEFAULT_ALGORITHMS",
    "DEFAULT_COMPARISONS",
    "DEFAULT_ENGINE",
    "DEFAULT_EVALS",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_N_OBJ",
    "DEFAULT_N_VAR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_POP_SIZE",
    "DEFAULT_PROBLEMS",
    "DEFAULT_SEEDS",
    "ROOT_DIR",
    "RunRecord",
    "build_config",
    "generate_markdown_report",
    "holm_adjust",
    "load_raw_records",
    "load_run_config",
    "metric_problem_name",
    "paired_wilcoxon_pvalue",
    "resolve_worker_count",
    "run_campaign",
    "run_once",
    "summarize_records",
    "write_analysis_artifacts",
    "write_csv",
    "write_run_config",
]
