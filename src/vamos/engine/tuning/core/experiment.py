from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .tuning_task import TuningTask, EvalContext
from vamos.engine.tuning.racing.random_search_tuner import RandomSearchTuner, TrialResult
from vamos.engine.tuning.racing.core import RacingTuner
from .scenario import Scenario
from .sampler import Sampler, UniformSampler, ModelBasedSampler
from .validation import (
    BenchmarkSuite,
    ConfigSpec,
    BenchmarkReport,
    ConfigSummary,
    StatisticalComparisonResult,
    run_benchmark_suite,
    summarize_benchmark,
    select_significantly_worse_configs,
)


class TunerKind(str, Enum):
    """
    Type of tuner used in a tuning experiment.
    """

    RANDOM = "random"
    RACING = "racing"


@dataclass
class ExperimentResult:
    """
    Full result of a tuning experiment, including tuning and optional validation.
    """

    name: str
    tuner_kind: TunerKind
    best_config: Dict[str, Any]
    tuning_history: List[TrialResult]
    benchmark_report: Optional[BenchmarkReport] = None
    benchmark_summaries: Optional[List[ConfigSummary]] = None
    benchmark_stats: Optional[StatisticalComparisonResult] = None


@dataclass
class TuningExperiment:
    """
    High-level orchestration of a tuning + validation experiment.

    This class wraps together:
    - a TuningTask,
    - a tuner kind (random or racing),
    - an optional Scenario (for racing),
    - a Sampler (Uniform or Model-based),
    - an optional BenchmarkSuite for validation,
    - optional baseline configurations for comparison.
    """

    name: str
    task: TuningTask
    tuner_kind: TunerKind = TunerKind.RANDOM

    # Random tuner settings
    max_trials: int = 50

    # Racing tuner settings
    scenario: Optional[Scenario] = None
    max_initial_configs: int = 20

    sampler: Optional[Sampler] = None

    validation_suite: Optional[BenchmarkSuite] = None
    baselines: Sequence[ConfigSpec] = field(default_factory=list)

    seed: int = 0
    maximize: bool = True

    def _create_sampler(self) -> Sampler:
        """
        Return the sampler to use for this experiment.
        """
        if self.sampler is not None:
            return self.sampler
        return UniformSampler(self.task.param_space)

    def _create_tuner(self, sampler: Sampler):
        """
        Create the underlying tuner object (random or racing).
        """
        if self.tuner_kind == TunerKind.RANDOM:
            return RandomSearchTuner(
                task=self.task,
                max_trials=self.max_trials,
                seed=self.seed,
                sampler=sampler,
            )

        if self.tuner_kind == TunerKind.RACING:
            if self.scenario is None:
                raise ValueError("Scenario must be provided when tuner_kind=RACING")
            return RacingTuner(
                task=self.task,
                scenario=self.scenario,
                seed=self.seed,
                max_initial_configs=self.max_initial_configs,
                sampler=sampler,
            )

        raise ValueError(f"Unsupported tuner_kind: {self.tuner_kind}")

    def run(
        self,
        eval_fn: Callable[[Dict[str, Any], EvalContext], float],
        verbose: bool = True,
    ) -> ExperimentResult:
        """
        Run the tuning experiment and optional validation.
        """
        sampler = self._create_sampler()
        tuner = self._create_tuner(sampler)

        if verbose:
            print(f"[experiment] Starting tuning: name={self.name}, tuner_kind={self.tuner_kind}")

        best_config, history = tuner.run(eval_fn, verbose=verbose)

        if verbose:
            print(f"[experiment] Tuning completed: {len(history)} trials")
            try:
                if history:
                    best_score = max(t.score for t in history) if self.maximize else min(t.score for t in history)
                    print(f"[experiment] Best tuning score: {best_score:.6f}")
            except Exception:
                pass

        benchmark_report: Optional[BenchmarkReport] = None
        benchmark_summaries: Optional[List[ConfigSummary]] = None
        benchmark_stats: Optional[StatisticalComparisonResult] = None

        if self.validation_suite is not None:
            if verbose:
                print(f"[experiment] Running validation on suite: {self.validation_suite.name}")

            configs_for_benchmark: List[ConfigSpec] = [
                ConfigSpec(label=f"{self.name} (tuned)", config=best_config)
            ]
            configs_for_benchmark.extend(self.baselines)

            benchmark_report = run_benchmark_suite(
                eval_fn=eval_fn,
                suite=self.validation_suite,
                configs=configs_for_benchmark,
                maximize=self.maximize,
            )
            benchmark_summaries = summarize_benchmark(
                report=benchmark_report,
                maximize=self.maximize,
            )
            try:
                benchmark_stats = select_significantly_worse_configs(
                    report=benchmark_report,
                    maximize=self.maximize,
                    alpha=0.05,
                )
            except Exception as exc:
                if verbose:
                    print(f"[experiment] Statistical comparison failed: {exc!r}")
                benchmark_stats = None

            if verbose and benchmark_summaries is not None:
                print("[experiment] Validation summary (rank 1 = best):")
                for s in benchmark_summaries:
                    print(
                        f"  rank={s.rank} label={s.label} "
                        f"mean={s.mean_score:.4f} std={s.std_score:.4f} "
                        f"min={s.min_score:.4f} max={s.max_score:.4f} runs={s.num_runs}"
                    )

            if verbose and benchmark_stats is not None:
                print(f"[experiment] Best config by mean score: {benchmark_stats.best_label}")
                print(f"[experiment] Statistically worse (alpha=0.05): {benchmark_stats.worse_labels}")
                print(f"[experiment] Not significantly worse: {benchmark_stats.non_worse_labels}")

        return ExperimentResult(
            name=self.name,
            tuner_kind=self.tuner_kind,
            best_config=best_config,
            tuning_history=history,
            benchmark_report=benchmark_report,
            benchmark_summaries=benchmark_summaries,
            benchmark_stats=benchmark_stats,
        )


def create_random_experiment(
    name: str,
    task: TuningTask,
    max_trials: int,
    seed: int = 0,
    sampler: Optional[Sampler] = None,
    validation_suite: Optional[BenchmarkSuite] = None,
    baselines: Optional[Sequence[ConfigSpec]] = None,
    maximize: bool = True,
) -> TuningExperiment:
    """
    Convenience factory to create a random-search tuning experiment.
    """
    return TuningExperiment(
        name=name,
        task=task,
        tuner_kind=TunerKind.RANDOM,
        max_trials=max_trials,
        scenario=None,
        max_initial_configs=0,
        sampler=sampler,
        validation_suite=validation_suite,
        baselines=list(baselines) if baselines is not None else [],
        seed=seed,
        maximize=maximize,
    )


def create_racing_experiment(
    name: str,
    task: TuningTask,
    scenario: Scenario,
    max_initial_configs: int,
    seed: int = 0,
    sampler: Optional[Sampler] = None,
    validation_suite: Optional[BenchmarkSuite] = None,
    baselines: Optional[Sequence[ConfigSpec]] = None,
    maximize: bool = True,
) -> TuningExperiment:
    """
    Convenience factory to create a racing-based tuning experiment.
    """
    return TuningExperiment(
        name=name,
        task=task,
        tuner_kind=TunerKind.RACING,
        max_trials=0,
        scenario=scenario,
        max_initial_configs=max_initial_configs,
        sampler=sampler,
        validation_suite=validation_suite,
        baselines=list(baselines) if baselines is not None else [],
        seed=seed,
        maximize=maximize,
    )


def example_experiment_usage() -> None:
    """
    Example of how to use TuningExperiment to run tuning + validation.
    This is a usage example and should not be executed on import.
    """
    # Placeholder example; replace with real objects in practice:
    # task: TuningTask = create_zdt1_tuning_task()
    # benchmark_suite: BenchmarkSuite = create_zdt_benchmark_suite()
    # eval_fn: Callable[[Dict[str, Any], EvalContext], float] = ...
    # default_cfg: Dict[str, Any] = ...
    # paper_cfg: Dict[str, Any] = ...
    # scenario = Scenario(max_experiments=1000, min_survivors=2, elimination_fraction=0.5, instance_order_random=True,
    #                    seed_order_random=True, start_instances=1, verbose=True, use_statistical_tests=True,
    #                    alpha=0.05, min_blocks_before_elimination=3, use_adaptive_budget=True,
    #                    initial_budget_per_run=5000, max_budget_per_run=20000, budget_growth_factor=2.0,
    #                    use_elitist_restarts=True, target_population_size=20, elite_fraction=0.3,
    #                    max_elite_archive_size=20, neighbor_fraction=0.5)
    #
    # baselines = [
    #     ConfigSpec(label="Default NSGA-II", config=default_cfg),
    #     ConfigSpec(label="AutoNSGA-II (paper)", config=paper_cfg),
    # ]
    #
    # sampler = ModelBasedSampler(
    #     param_space=task.param_space,
    #     exploration_prob=0.2,
    #     min_samples_to_model=5,
    # )
    #
    # experiment = create_racing_experiment(
    #     name="zdt1_racing_autonsga",
    #     task=task,
    #     scenario=scenario,
    #     max_initial_configs=20,
    #     seed=42,
    #     sampler=sampler,
    #     validation_suite=benchmark_suite,
    #     baselines=baselines,
    #     maximize=True,
    # )
    #
    # result = experiment.run(eval_fn, verbose=True)
    # print("Best tuned config:")
    # print(result.best_config)


__all__ = [
    "TunerKind",
    "ExperimentResult",
    "TuningExperiment",
    "create_random_experiment",
    "create_racing_experiment",
]
