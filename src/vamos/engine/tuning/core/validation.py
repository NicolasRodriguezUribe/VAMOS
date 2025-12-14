from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .tuning_task import Instance, EvalContext
from .stats import _z_critical


@dataclass
class BenchmarkSuite:
    """
    Benchmark suite for validating algorithm configurations.

    A suite defines:
    - a set of problem instances,
    - a set of random seeds,
    - a fixed evaluation budget per run.

    All configurations will be evaluated on all (instance, seed) combinations
    using the same eval_fn and budget.
    """

    name: str
    instances: Sequence[Instance]
    seeds: Sequence[int]
    budget_per_run: int

    def __post_init__(self) -> None:
        if not self.instances:
            raise ValueError("BenchmarkSuite.instances must not be empty")
        if not self.seeds:
            raise ValueError("BenchmarkSuite.seeds must not be empty")
        if self.budget_per_run <= 0:
            raise ValueError("BenchmarkSuite.budget_per_run must be positive")


@dataclass
class ConfigSpec:
    """
    Configuration specification for benchmarking.

    Attributes:
        label: Human-readable identifier (e.g., 'Default NSGA-II',
            'AutoNSGA-II (paper)', 'Racing-tuned best').
        config: Dictionary of hyperparameters / algorithm settings, to be
            passed to eval_fn together with an EvalContext.
    """

    label: str
    config: Mapping[str, Any]


@dataclass
class RunResult:
    """
    Result of a single run of a configuration on a specific instance and seed.
    """

    config_label: str
    instance_name: str
    seed: int
    run_index: int
    score: float


@dataclass
class BenchmarkReport:
    """
    Full report of a benchmark run over a suite and a set of configurations.
    """

    suite: BenchmarkSuite
    configs: Sequence[ConfigSpec]
    results: List[RunResult]


@dataclass
class ConfigSummary:
    """
    Summary statistics of a configuration over all runs in a benchmark report.
    """

    label: str
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    num_runs: int
    rank: int


@dataclass
class ConfigInstanceSummary:
    """
    Summary statistics for a configuration on a specific instance.
    """

    config_label: str
    instance_name: str
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    num_runs: int


@dataclass
class StatisticalComparisonResult:
    """
    Statistical comparison of configurations vs the best one.
    """

    best_label: str
    worse_labels: List[str]
    non_worse_labels: List[str]


def run_benchmark_suite(
    eval_fn: Callable[[Dict[str, Any], EvalContext], float],
    suite: BenchmarkSuite,
    configs: Sequence[ConfigSpec],
    maximize: bool = True,
) -> BenchmarkReport:
    """
    Evaluate a set of configurations on a benchmark suite.
    """
    results: List[RunResult] = []

    for cfg_spec in configs:
        cfg_dict = dict(cfg_spec.config)
        for instance in suite.instances:
            for seed in suite.seeds:
                ctx = EvalContext(instance=instance, seed=seed, budget=suite.budget_per_run)
                score = eval_fn(cfg_dict, ctx)
                result = RunResult(
                    config_label=cfg_spec.label,
                    instance_name=getattr(instance, "name", str(instance)),
                    seed=seed,
                    run_index=0,
                    score=float(score),
                )
                results.append(result)

    return BenchmarkReport(suite=suite, configs=list(configs), results=results)


def summarize_benchmark(
    report: BenchmarkReport,
    maximize: bool = True,
) -> List[ConfigSummary]:
    """
    Compute summary statistics per configuration over all runs in the report.
    """
    scores_by_label: Dict[str, List[float]] = {}
    for r in report.results:
        scores_by_label.setdefault(r.config_label, []).append(r.score)

    # Compute average rank across runs to break ties fairly.
    rank_accum: Dict[str, List[int]] = {label: [] for label in scores_by_label}
    # group results by (instance, seed) to rank within each scenario
    grouped: Dict[Tuple[str, int], List[RunResult]] = {}
    for r in report.results:
        grouped.setdefault((r.instance_name, r.seed), []).append(r)
    for runs in grouped.values():
        # sort by score
        runs_sorted = sorted(runs, key=lambda rr: rr.score, reverse=maximize)
        for idx, rr in enumerate(runs_sorted, start=1):
            rank_accum[rr.config_label].append(idx)

    summaries: List[ConfigSummary] = []
    avg_rank_map: Dict[str, float] = {}
    for label, scores in scores_by_label.items():
        arr = np.asarray(scores, dtype=float)
        mean = float(arr.mean()) if arr.size > 0 else float("nan")
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        min_score = float(arr.min()) if arr.size > 0 else float("nan")
        max_score = float(arr.max()) if arr.size > 0 else float("nan")
        avg_rank = float(np.mean(rank_accum.get(label, [np.inf])))
        avg_rank_map[label] = avg_rank
        summaries.append(
            ConfigSummary(
                label=label,
                mean_score=mean,
                std_score=std,
                min_score=min_score,
                max_score=max_score,
                num_runs=int(arr.size),
                rank=-1,
            )
        )
        summaries[-1].avg_rank = avg_rank  # type: ignore[attr-defined]

    if not summaries:
        return []

    def _sort_key(summary: ConfigSummary) -> Tuple[float, float, float]:
        avg_rank = avg_rank_map.get(summary.label, np.inf)
        best_score = summary.max_score
        mean_score = summary.mean_score
        if not np.isfinite(best_score):
            best_score = -np.inf if maximize else np.inf
        if not np.isfinite(mean_score):
            mean_score = -np.inf if maximize else np.inf
        if maximize:
            return (avg_rank, -best_score, -mean_score)
        return (avg_rank, best_score, mean_score)

    order = sorted(range(len(summaries)), key=lambda idx: _sort_key(summaries[idx]))

    ranked_summaries: List[ConfigSummary] = []
    for rank_idx, idx in enumerate(order, start=1):
        s = summaries[int(idx)]
        s.rank = rank_idx
        ranked_summaries.append(s)

    return ranked_summaries


def summarize_benchmark_per_instance(
    report: BenchmarkReport,
) -> List[ConfigInstanceSummary]:
    """
    Compute per-instance summary statistics for each configuration.
    """
    scores_map: Dict[Tuple[str, str], List[float]] = {}

    for r in report.results:
        key = (r.config_label, r.instance_name)
        scores_map.setdefault(key, []).append(r.score)

    summaries: List[ConfigInstanceSummary] = []
    for (label, inst_name), scores in scores_map.items():
        arr = np.asarray(scores, dtype=float)
        mean = float(arr.mean()) if arr.size > 0 else float("nan")
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        min_score = float(arr.min()) if arr.size > 0 else float("nan")
        max_score = float(arr.max()) if arr.size > 0 else float("nan")
        summaries.append(
            ConfigInstanceSummary(
                config_label=label,
                instance_name=inst_name,
                mean_score=mean,
                std_score=std,
                min_score=min_score,
                max_score=max_score,
                num_runs=int(arr.size),
            )
        )

    return summaries


def select_significantly_worse_configs(
    report: BenchmarkReport,
    maximize: bool = True,
    alpha: float = 0.05,
) -> StatisticalComparisonResult:
    """
    Perform a paired test vs the best configuration to decide which configs
    are significantly worse, using all (instance, seed) combinations as blocks.
    """
    labels = sorted({c.label for c in report.configs})
    if not labels:
        raise ValueError("No configurations in report")

    score_map: Dict[Tuple[str, str, int], float] = {}
    for r in report.results:
        key = (r.config_label, r.instance_name, r.seed)
        score_map[key] = r.score

    blocks = {(r.instance_name, r.seed) for r in report.results}
    common_blocks: List[Tuple[str, int]] = []
    for inst_name, seed in blocks:
        if all((label, inst_name, seed) in score_map for label in labels):
            common_blocks.append((inst_name, seed))

    if not common_blocks:
        raise ValueError("No common blocks across all configurations for statistical comparison")

    n_configs = len(labels)
    n_blocks = len(common_blocks)
    scores = np.zeros((n_configs, n_blocks), dtype=float)

    for i, label in enumerate(labels):
        for j, (inst_name, seed) in enumerate(common_blocks):
            key = (label, inst_name, seed)
            scores[i, j] = score_map[key]

    mean_scores = scores.mean(axis=1)
    best_idx = int(np.argmax(mean_scores)) if maximize else int(np.argmin(mean_scores))
    best_label = labels[best_idx]
    best_scores = scores[best_idx, :]

    z_crit = _z_critical(alpha)

    worse_labels: List[str] = []
    non_worse_labels: List[str] = []

    for i, label in enumerate(labels):
        if i == best_idx:
            non_worse_labels.append(label)
            continue

        cfg_scores = scores[i, :]
        diffs = best_scores - cfg_scores if maximize else cfg_scores - best_scores

        mean_diff = float(diffs.mean())
        sd_diff = float(diffs.std(ddof=1)) if n_blocks > 1 else 0.0

        if sd_diff <= 1e-12:
            if mean_diff > 0.0:
                worse_labels.append(label)
            else:
                non_worse_labels.append(label)
            continue

        t_stat = mean_diff / (sd_diff / np.sqrt(n_blocks))

        if t_stat > z_crit:
            worse_labels.append(label)
        else:
            non_worse_labels.append(label)

    return StatisticalComparisonResult(
        best_label=best_label,
        worse_labels=worse_labels,
        non_worse_labels=non_worse_labels,
    )


def example_benchmark_usage() -> None:
    """
    Example of how to run a benchmark suite comparing multiple configurations.

    This is a usage example and should not be executed on import.
    """
    # Example placeholders; replace with real objects in practice:
    # instances: Sequence[Instance] = [...]
    # eval_fn: Callable[[Dict[str, Any], EvalContext], float] = ...
    # config_default, config_paper, config_tuned: Dict[str, Any] = ...
    #
    # suite = BenchmarkSuite(
    #     name="ZDT1-3_30vars",
    #     instances=instances,
    #     seeds=[1, 2, 3, 4, 5],
    #     budget_per_run=20000,
    # )
    #
    # configs = [
    #     ConfigSpec(label="Default NSGA-II", config=config_default),
    #     ConfigSpec(label="AutoNSGA-II (paper)", config=config_paper),
    #     ConfigSpec(label="Racing-tuned best", config=config_tuned),
    # ]
    #
    # report = run_benchmark_suite(eval_fn, suite, configs, maximize=True)
    # summaries = summarize_benchmark(report, maximize=True)
    #
    # print("Benchmark summary (best rank = 1):")
    # for s in summaries:
    #     print(
    #         f"{s.rank}. {s.label}: mean={s.mean_score:.4f}, std={s.std_score:.4f}, "
    #         f"min={s.min_score:.4f}, max={s.max_score:.4f}, runs={s.num_runs}"
    #     )
    #
    # comp = select_significantly_worse_configs(report, maximize=True, alpha=0.05)
    # print(f"Best configuration by mean score: {comp.best_label}")
    # print("Statistically worse configs:", comp.worse_labels)
    # print("Not significantly worse:", comp.non_worse_labels)


__all__ = [
    "BenchmarkSuite",
    "ConfigSpec",
    "RunResult",
    "BenchmarkReport",
    "ConfigSummary",
    "ConfigInstanceSummary",
    "StatisticalComparisonResult",
    "run_benchmark_suite",
    "summarize_benchmark",
    "summarize_benchmark_per_instance",
    "select_significantly_worse_configs",
    "example_benchmark_usage",
]
