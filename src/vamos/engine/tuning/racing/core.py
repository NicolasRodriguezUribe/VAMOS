from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .scenario import Scenario
from .tuning_task import TuningTask, Instance, EvalContext
from .random_search_tuner import TrialResult
from .param_space import ParamSpace
from .sampler import Sampler, UniformSampler, ModelBasedSampler
from .state import ConfigState, EliteEntry
from .schedule import build_schedule
from .elimination import eliminate_configs, update_elite_archive
from .refill import refill_population


class RacingTuner:
    """
    Irace-inspired racing tuner for algorithm configuration.

    This tuner evaluates configurations over instances and seeds in stages.
    After each stage, it eliminates clearly inferior configurations based on
    aggregated performance, optionally using statistical paired tests against
    the current best configuration, until a small set of survivors remains or
    the experiment budget is exhausted.
    """

    def __init__(
        self,
        task: TuningTask,
        scenario: Scenario,
        seed: int = 0,
        max_initial_configs: int = 20,
        sampler: Optional[Sampler] = None,
    ) -> None:
        self.task = task
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.max_initial_configs = max_initial_configs
        self._stage_index: int = 0
        self._elite_archive: List[EliteEntry] = []
        self._next_config_id: int = self.max_initial_configs

        self.param_space: ParamSpace = task.param_space
        self.instances: Sequence[Instance] = list(task.instances)
        self.seeds: Sequence[int] = list(task.seeds)

        if sampler is None:
            self.sampler: Sampler = UniformSampler(self.param_space)
        else:
            self.sampler = sampler

        self._schedule: List[Tuple[int, int]] = build_schedule(
            self.instances,
            self.seeds,
            start_instances=self.scenario.start_instances,
            instance_order_random=self.scenario.instance_order_random,
            seed_order_random=self.scenario.seed_order_random,
            rng=self.rng,
        )

    def _sample_initial_configs(self) -> List[ConfigState]:
        """Sample the initial population of configurations."""
        configs: List[ConfigState] = []
        for config_id in range(self.max_initial_configs):
            cfg = self.sampler.sample(self.rng)
            state = ConfigState(config_id=config_id, config=cfg, alive=True)
            configs.append(state)
        return configs

    def _current_budget(self) -> int:
        """
        Compute the evaluation budget for the current stage, respecting adaptive budget settings and task-level caps.
        """
        base_budget = self.task.budget_per_run
        if self.scenario.use_adaptive_budget:
            if self.scenario.initial_budget_per_run is not None:
                base_budget = self.scenario.initial_budget_per_run

            budget = int(round(base_budget * (self.scenario.budget_growth_factor ** self._stage_index)))

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
        eval_fn: Callable[[Dict[str, Any], EvalContext], float],
        verbose: Optional[bool] = None,
    ) -> Tuple[Dict[str, Any], List[TrialResult]]:
        """
        Run the racing process and return best_config and the evaluation history.
        """
        verbose_flag = self.scenario.verbose if verbose is None else verbose

        schedule = list(self._schedule)
        configs: List[ConfigState] = self._sample_initial_configs()
        num_experiments = 0

        for inst_idx, seed_idx in schedule:
            if self.scenario.max_stages is not None and self._stage_index >= self.scenario.max_stages:
                if verbose_flag:
                    print("[racing] Reached maximum number of stages.")
                break

            if self._num_alive(configs) == 0:
                break

            stage_alive = self._num_alive(configs)
            if num_experiments + stage_alive > self.scenario.max_experiments:
                if verbose_flag:
                    print("[racing] Experiment budget exhausted before next stage.")
                break

            if verbose_flag:
                print(
                    f"[racing] Stage {self._stage_index}: instance {inst_idx}, seed idx {seed_idx}, alive={stage_alive}"
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

            reached_budget = num_experiments >= self.scenario.max_experiments
            reached_min_survivors = self._num_alive(configs) <= self.scenario.min_survivors

            self._stage_index += 1

            if reached_budget:
                if verbose_flag:
                    print("[racing] Reached maximum experiment budget.")
                break

            if reached_min_survivors:
                if verbose_flag:
                    print("[racing] Reached minimum survivors, stopping early.")
                break

        best_state, history = self._finalize_results(configs)
        if best_state is None:
            raise RuntimeError("RacingTuner finished without a valid configuration.")

        if verbose_flag and best_state.score is not None:
            print(f"[racing] Best score={best_state.score:.6f} after stage {self._stage_index}.")

        return best_state.config, history

    def _run_stage(
        self,
        configs: List[ConfigState],
        inst_idx: int,
        seed_idx: int,
        eval_fn: Callable[[Dict[str, Any], EvalContext], float],
    ) -> None:
        """Evaluate all alive configurations on a given (instance, seed)."""
        instance = self.instances[inst_idx]
        seed = self.seeds[seed_idx]
        budget = self._current_budget()

        for state in configs:
            if not state.alive:
                continue
            ctx = EvalContext(instance=instance, seed=seed, budget=budget)
            score = float(eval_fn(state.config, ctx))
            state.scores.append(score)

    def _count_new_experiments(self, configs: List[ConfigState]) -> int:
        """Count how many alive configs were evaluated in the last stage."""
        return self._num_alive(configs)

    def _num_alive(self, configs: List[ConfigState]) -> int:
        return sum(1 for c in configs if c.alive)

    def _finalize_results(self, configs: List[ConfigState]) -> Tuple[Optional[EliteEntry], List[TrialResult]]:
        history: List[TrialResult] = []
        best_state: Optional[EliteEntry] = None
        best_score: Optional[float] = None

        for state in configs:
            if not state.scores:
                agg_score = float("nan")
            else:
                agg_score = float(self.task.aggregator(state.scores))

            history.append(
                TrialResult(
                    trial_id=state.config_id,
                    config=state.config,
                    score=agg_score,
                    details={"num_evals": len(state.scores), "alive": state.alive},
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
