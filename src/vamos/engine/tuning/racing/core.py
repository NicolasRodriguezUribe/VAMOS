from __future__ import annotations

import logging
from typing import Any
from collections.abc import Sequence

import numpy as np

from .eval_types import EvalFn
from .scenario import Scenario
from .tuning_task import TuningTask, Instance, EvalContext
from .random_search_tuner import TrialResult
from .param_space import ParamSpace
from .sampler import Sampler, UniformSampler, ModelBasedSampler
from .state import ConfigState, EliteEntry
from .schedule import build_schedule
from .elimination import eliminate_configs, update_elite_archive
from .refill import refill_population
from .multi_fidelity import _run_multi_fidelity


from joblib import Parallel, delayed  # type: ignore[import-untyped]


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _eval_worker(eval_fn: EvalFn, config: dict[str, Any], ctx: EvalContext) -> float:
    result = eval_fn(config, ctx)
    if isinstance(result, tuple):
        return float(result[0])
    return float(result)


class RacingTuner:
    """Racing tuner for algorithm configuration."""

    def __init__(
        self,
        task: TuningTask,
        scenario: Scenario,
        seed: int = 0,
        max_initial_configs: int = 20,
        sampler: Sampler | None = None,
        initial_configs: list[dict[str, Any]] | None = None,
    ) -> None:
        self.task = task
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.max_initial_configs = max_initial_configs
        self._stage_index: int = 0
        self._elite_archive: list[EliteEntry] = []
        self._next_config_id: int = 0  # Re-indexed below
        self._best_score_history: list[float] = []  # For convergence detection

        # Injection of default/user configurations
        self.initial_configs_payload = initial_configs or []

        self.param_space: ParamSpace = task.param_space
        self.instances: Sequence[Instance] = list(task.instances)
        self.seeds: Sequence[int] = list(task.seeds)

        if sampler is None:
            self.sampler: Sampler = UniformSampler(self.param_space)
        else:
            self.sampler = sampler

        self._schedule: list[tuple[int, int]] = build_schedule(
            self.instances,
            self.seeds,
            start_instances=self.scenario.start_instances,
            instance_order_random=self.scenario.instance_order_random,
            seed_order_random=self.scenario.seed_order_random,
            rng=self.rng,
        )

    def _sample_initial_configs(self) -> list[ConfigState]:
        configs: list[ConfigState] = []
        config_id_counter = 0

        # 1. Add injected configs
        for user_cfg in self.initial_configs_payload:
            state = ConfigState(config_id=config_id_counter, config=user_cfg, alive=True)
            configs.append(state)
            config_id_counter += 1

        # 2. Sample remainder
        needed = max(0, self.max_initial_configs - len(configs))
        for _ in range(needed):
            cfg = self.sampler.sample(self.rng)
            state = ConfigState(config_id=config_id_counter, config=cfg, alive=True)
            configs.append(state)
            config_id_counter += 1

        self._next_config_id = config_id_counter
        return configs

    def _current_budget(self) -> int:
        base_budget = self.task.budget_per_run
        if self.scenario.use_adaptive_budget:
            if self.scenario.initial_budget_per_run is not None:
                base_budget = self.scenario.initial_budget_per_run

            budget = int(round(base_budget * (self.scenario.budget_growth_factor**self._stage_index)))

            if self.scenario.max_budget_per_run is not None:
                budget = min(budget, self.scenario.max_budget_per_run)

            if self.task.budget_per_run is not None:
                budget = min(budget, self.task.budget_per_run)

            if budget <= 0:
                budget = max(1, self.task.budget_per_run or 1)
            return budget

        if base_budget is None or base_budget <= 0:
            raise ValueError("task.budget_per_run must be positive when adaptive budget is disabled")
        return base_budget

    def run(
        self,
        eval_fn: EvalFn,
        verbose: bool | None = None,
    ) -> tuple[dict[str, Any], list[TrialResult]]:
        # Dispatch to multi-fidelity if enabled
        if self.scenario.use_multi_fidelity:
            return _run_multi_fidelity(self, eval_fn, verbose)

        verbose_flag = self.scenario.verbose if verbose is None else verbose

        schedule = list(self._schedule)
        configs: list[ConfigState] = self._sample_initial_configs()
        num_experiments = 0

        for inst_idx, seed_idx in schedule:
            if self.scenario.max_stages is not None and self._stage_index >= self.scenario.max_stages:
                if verbose_flag:
                    _logger().info("[racing] Reached maximum number of stages.")
                break

            if self._num_alive(configs) == 0:
                break

            stage_alive = self._num_alive(configs)
            if num_experiments + stage_alive > self.scenario.max_experiments:
                if verbose_flag:
                    _logger().info("[racing] Experiment budget exhausted before next stage.")
                break

            if verbose_flag:
                _logger().info(
                    "[racing] Stage %s: instance %s, seed idx %s, alive=%s",
                    self._stage_index,
                    inst_idx,
                    seed_idx,
                    stage_alive,
                )

            self._run_stage(configs, inst_idx, seed_idx, eval_fn)
            stage_eval_count = self._count_new_experiments(configs)
            num_experiments += stage_eval_count

            eliminated_any = eliminate_configs(configs, task=self.task, scenario=self.scenario)

            if eliminated_any:
                self._elite_archive = update_elite_archive(
                    configs,
                    task=self.task,
                    scenario=self.scenario,
                    elite_archive=self._elite_archive,
                )
                if self.scenario.use_elitist_restarts:
                    self._next_config_id = refill_population(
                        configs,
                        scenario=self.scenario,
                        param_space=self.param_space,
                        sampler=self.sampler,
                        elite_archive=self._elite_archive,
                        target_population_size=self.scenario.target_population_size or self.max_initial_configs,
                        rng=self.rng,
                        next_config_id=self._next_config_id,
                    )
                if isinstance(self.sampler, ModelBasedSampler):
                    survivor_configs = [c.config for c in configs if c.alive]
                    self.sampler.update(survivor_configs)

            # Track best score for convergence detection
            current_best = self._get_current_best_score(configs)
            if current_best is not None:
                self._best_score_history.append(current_best)

            reached_budget = num_experiments >= self.scenario.max_experiments
            reached_min_survivors = self._num_alive(configs) <= self.scenario.min_survivors
            reached_convergence = self._check_convergence()

            self._stage_index += 1

            if reached_budget:
                if verbose_flag:
                    _logger().info("[racing] Reached maximum experiment budget.")
                break

            if reached_min_survivors:
                if verbose_flag:
                    _logger().info("[racing] Reached minimum survivors, stopping early.")
                break

            if reached_convergence:
                if verbose_flag:
                    _logger().info("[racing] Converged after %s stages (no improvement).", self._stage_index)
                break

        best_state, history = self._finalize_results(configs)
        if best_state is None:
            raise RuntimeError("RacingTuner finished without a valid configuration.")

        if verbose_flag and best_state.score is not None:
            _logger().info(
                "[racing] Best score=%.6f after stage %s.",
                best_state.score,
                self._stage_index,
            )

        return best_state.config, history

    def _run_stage(
        self,
        configs: list[ConfigState],
        inst_idx: int,
        seed_idx: int,
        eval_fn: EvalFn,
    ) -> None:
        instance = self.instances[inst_idx]
        seed = self.seeds[seed_idx]
        budget = self._current_budget()

        # Identify jobs to run
        tasks = []
        indices = []
        for idx, state in enumerate(configs):
            if not state.alive:
                continue
            ctx = EvalContext(instance=instance, seed=seed, budget=budget)
            tasks.append((state.config, ctx))
            indices.append(idx)

        if not tasks:
            return

        if self.scenario.n_jobs == 1:
            # Sequential execution (avoid overhead)
            for i, (cfg, ctx) in enumerate(tasks):
                result = eval_fn(cfg, ctx)
                score = float(result[0]) if isinstance(result, tuple) else float(result)
                configs[indices[i]].scores.append(score)
        else:
            # Parallel execution with joblib
            results = Parallel(n_jobs=self.scenario.n_jobs)(delayed(_eval_worker)(eval_fn, cfg, ctx) for cfg, ctx in tasks)

            for i, score in enumerate(results):
                configs[indices[i]].scores.append(score)

    def _count_new_experiments(self, configs: list[ConfigState]) -> int:
        """Count how many alive configs were evaluated in the last stage."""
        return self._num_alive(configs)

    def _num_alive(self, configs: list[ConfigState]) -> int:
        return sum(1 for c in configs if c.alive)

    def _get_current_best_score(self, configs: list[ConfigState]) -> float | None:
        """Get the current best aggregated score among alive configs."""
        best: float | None = None
        for state in configs:
            if not state.alive or not state.scores:
                continue
            agg = float(self.task.aggregator(state.scores))
            if best is None:
                best = agg
            elif self.task.maximize and agg > best:
                best = agg
            elif not self.task.maximize and agg < best:
                best = agg
        return best

    def _check_convergence(self) -> bool:
        """Check if best score has stagnated for convergence_window stages."""
        window = self.scenario.convergence_window
        if window <= 0:
            return False  # Disabled

        history = self._best_score_history
        if len(history) < window:
            return False

        # Get scores over the last 'window' stages
        recent = history[-window:]
        oldest = recent[0]
        newest = recent[-1]

        # Compute relative improvement
        if abs(oldest) < 1e-12:
            # Avoid division by zero; if scores are near zero, use absolute diff
            improvement = abs(newest - oldest)
        else:
            improvement = abs((newest - oldest) / oldest)

        return improvement < self.scenario.convergence_threshold

    def _finalize_results(self, configs: list[ConfigState]) -> tuple[EliteEntry | None, list[TrialResult]]:
        history: list[TrialResult] = []
        best_state: EliteEntry | None = None
        best_score: float | None = None

        for state in configs:
            scores = state.scores
            details = {"num_evals": len(state.scores), "alive": state.alive}
            if self.scenario.use_multi_fidelity:
                final_level = len(self.scenario.fidelity_levels) - 1
                final_scores = state.fidelity_scores.get(final_level) or []
                details["final_fidelity_level"] = int(final_level)
                details["num_final_evals"] = int(len(final_scores))
                if final_scores:
                    scores = final_scores

            if not scores:
                agg_score = float("nan")
            else:
                agg_score = float(self.task.aggregator(scores))

            history.append(
                TrialResult(
                    trial_id=state.config_id,
                    config=state.config,
                    score=agg_score,
                    details=details,
                )
            )

            if not state.alive or not state.scores:
                continue

            if best_score is None:
                best_state = EliteEntry(config=state.config, score=agg_score)
                best_score = agg_score
                continue

            if self.task.maximize and agg_score > best_score:
                best_state = EliteEntry(config=state.config, score=agg_score)
                best_score = agg_score
            elif not self.task.maximize and agg_score < best_score:
                best_state = EliteEntry(config=state.config, score=agg_score)
                best_score = agg_score

        return best_state, history


__all__ = ["RacingTuner"]
