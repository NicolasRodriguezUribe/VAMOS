from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .scenario import Scenario
from .tuning_task import TuningTask, Instance, EvalContext
from .random_search_tuner import TrialResult
from .param_space import ParamSpace, Real, Int, Categorical
from .stats import build_score_matrix, select_configs_by_paired_test
from .sampler import Sampler, UniformSampler, ModelBasedSampler


@dataclass
class ConfigState:
    """
    Internal structure to keep track of a single configuration during racing.
    """

    config_id: int
    config: Dict[str, Any]
    alive: bool = True
    # Scores are stored in the same order as the evaluation schedule.
    scores: List[float] = field(default_factory=list)


@dataclass
class EliteEntry:
    """
    Elite archive entry: configuration and its aggregated score.
    """

    config: Dict[str, Any]
    score: float


class RacingTuner:
    """
    Irace-inspired racing tuner for algorithm configuration.

    This tuner evaluates configurations over instances and seeds in stages.
    After each stage, it eliminates clearly inferior configurations based on
    aggregated performance, optionally using statistical paired tests against
    the current best configuration, until a small set of survivors remains or
    the experiment budget is exhausted. Instance coverage is multi-stage
    (early stages use a subset of instances), evaluation budgets can grow
    adaptively across stages if enabled, sampling is pluggable via a
    Sampler (uniform by default, model-based optional), and optional elitist
    restarts can refill the population via local search around top configs.
    This is a synchronous, rank-based F-race skeleton with an optional
    statistical decision rule.
    """

    def __init__(
        self,
        task: TuningTask,
        scenario: Scenario,
        seed: int = 0,
        max_initial_configs: int = 20,
        sampler: Optional[Sampler] = None,
    ) -> None:
        """
        Args:
            task: TuningTask describing param space, instances, seeds, etc.
            scenario: Racing scenario settings controlling the racing strategy.
            seed: RNG seed for sampling configurations and instance/seed ordering.
            max_initial_configs: Number of configurations to sample initially.
            sampler: Strategy to sample configurations. Defaults to uniform sampling.
        """
        self.task = task
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.max_initial_configs = max_initial_configs
        self._stage_index: int = 0
        """
        Current stage index (0-based). Incremented after each elimination step
        that actually removes at least one configuration.
        """
        self._elite_archive: List[EliteEntry] = []
        self._next_config_id: int = self.max_initial_configs
        """
        _next_config_id assigns unique IDs to newly spawned configurations after
        the initial sampling phase.
        """

        self.param_space: ParamSpace = task.param_space
        self.instances: Sequence[Instance] = list(task.instances)
        self.seeds: Sequence[int] = list(task.seeds)

        if sampler is None:
            self.sampler: Sampler = UniformSampler(self.param_space)
        else:
            self.sampler = sampler

        self._schedule: List[Tuple[int, int]] = self._build_schedule()

    def _build_schedule(self) -> List[Tuple[int, int]]:
        """
        Build a list of (instance_index, seed_index) pairs that defines the
        evaluation order for the race. Respects scenario.instance_order_random
        and scenario.seed_order_random. The schedule may be truncated later
        by max_experiments or max_stages.
        """
        n_instances = len(self.instances)
        n_seeds = len(self.seeds)
        if n_instances == 0 or n_seeds == 0:
            return []

        inst_indices = list(range(n_instances))
        seed_indices = list(range(n_seeds))

        if self.scenario.instance_order_random:
            self.rng.shuffle(inst_indices)
        if self.scenario.seed_order_random:
            self.rng.shuffle(seed_indices)

        k = max(1, min(self.scenario.start_instances, n_instances))
        stage1_instances = inst_indices[:k]
        remaining_instances = inst_indices[k:]

        schedule: List[Tuple[int, int]] = []

        # Stage 1: restricted set of instances
        for seed_idx in seed_indices:
            for inst_idx in stage1_instances:
                schedule.append((inst_idx, seed_idx))

        # Stage 2+: remaining instances
        if remaining_instances:
            for seed_idx in seed_indices:
                for inst_idx in remaining_instances:
                    schedule.append((inst_idx, seed_idx))

        return schedule

    def _sample_initial_configs(self) -> List[ConfigState]:
        """
        Sample max_initial_configs configurations from the param space and
        return them as ConfigState objects.
        """
        configs: List[ConfigState] = []
        for config_id in range(self.max_initial_configs):
            cfg = self.sampler.sample(self.rng)
            state = ConfigState(config_id=config_id, config=cfg, alive=True)
            configs.append(state)
        return configs

    def _compute_aggregated_scores(
        self,
        configs: List[ConfigState],
    ) -> Tuple[List[int], np.ndarray]:
        """
        Compute aggregated scores for all alive configs that have at least one score.

        Returns:
            indices: indices into `configs` for which aggregated scores exist.
            agg_scores: aggregated scores in the same order as indices.
        """
        scores, alive_indices = build_score_matrix(configs)
        if scores.size == 0 or len(alive_indices) == 0:
            return [], np.array([], dtype=float)

        agg_values = self._aggregate_rows(scores)
        return alive_indices, np.asarray(agg_values, dtype=float)

    def _current_budget(self) -> int:
        """
        Compute the evaluation budget for the current stage, respecting adaptive
        budget settings and task-level caps.
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
        Run the racing process and return:

            best_config, history

        where best_config is the configuration dictionary of the best surviving
        configuration at the end, and history is a list of TrialResult objects
        with all evaluated configurations and their final aggregated scores.

        The `eval_fn` is the same used in the base Tuner: it takes a config and
        an EvalContext (instance, seed, budget) and returns a scalar score
        (higher is better if task.maximize=True).
        """
        verbose_flag = self.scenario.verbose if verbose is None else verbose

        schedule = list(self._schedule)
        configs: List[ConfigState] = self._sample_initial_configs()
        num_experiments = 0

        for inst_idx, seed_idx in schedule:
            if self._num_alive(configs) == 0:
                break

            if self.scenario.max_stages is not None and self._stage_index >= self.scenario.max_stages:
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

            if num_experiments >= self.scenario.max_experiments:
                if verbose_flag:
                    print("[racing] Reached maximum experiment budget.")
                break

            eliminated_any = self._eliminate_configs(configs)

            if eliminated_any:
                self._stage_index += 1
                self._update_elite_archive(configs)
                self._refill_population(configs)
                if isinstance(self.sampler, ModelBasedSampler):
                    survivor_configs = [c.config for c in configs if c.alive]
                    self.sampler.update(survivor_configs)

            if self._num_alive(configs) <= self.scenario.min_survivors:
                if verbose_flag:
                    print("[racing] Reached minimum survivors, stopping early.")
                break

            if self.scenario.max_stages is not None and self._stage_index >= self.scenario.max_stages:
                if verbose_flag:
                    print("[racing] Reached maximum number of stages.")
                break

        history: List[TrialResult] = []
        best_state: Optional[ConfigState] = None
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
                best_state = state
                best_score = agg_score
                continue

            if self.task.maximize and agg_score > best_score:
                best_state = state
                best_score = agg_score
            elif not self.task.maximize and agg_score < best_score:
                best_state = state
                best_score = agg_score

        if best_state is None or best_score is None:
            raise RuntimeError("RacingTuner finished without a valid configuration.")

        if verbose_flag:
            print(f"[racing] Best score={best_score:.6f} after stage {self._stage_index}.")

        return best_state.config, history

    def _run_stage(
        self,
        configs: List[ConfigState],
        inst_idx: int,
        seed_idx: int,
        eval_fn: Callable[[Dict[str, Any], EvalContext], float],
    ) -> None:
        """
        Evaluate all alive configurations on a given (instance, seed), using the
        budget determined by the current stage (adaptive budget if enabled).

        Appends the resulting score to each config's scores list in the same
        order as schedule progression (i.e., scores[t] correspond to
        schedule[t] for that config).
        """
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

    def _eliminate_configs(self, configs: List[ConfigState]) -> bool:
        """
        Eliminate configurations based on the current scores.

        If statistical tests are enabled and enough blocks have been evaluated,
        perform paired tests against the current best configuration. Otherwise,
        fall back to rank-based elimination.

        Returns:
            True if at least one configuration was eliminated.
        """
        scores, alive_indices = build_score_matrix(configs)
        if scores.size == 0 or len(alive_indices) <= 1:
            return False

        _, n_blocks = scores.shape

        if (
            not self.scenario.use_statistical_tests
            or n_blocks < self.scenario.min_blocks_before_elimination
        ):
            return self._rank_based_elimination(configs, scores, alive_indices)

        keep_mask = select_configs_by_paired_test(
            scores=scores,
            maximize=self.task.maximize,
            alpha=self.scenario.alpha,
        )

        num_keep = int(keep_mask.sum())
        if num_keep <= 0:
            agg_scores = self._aggregate_rows(scores)
            best_idx = int(np.argmax(agg_scores)) if self.task.maximize else int(np.argmin(agg_scores))
            keep_mask[best_idx] = True
            num_keep = 1

        if num_keep < self.scenario.min_survivors:
            return self._force_keep_top_k(configs, scores, alive_indices, self.scenario.min_survivors)

        eliminated_any = False
        for row_idx, cfg_idx in enumerate(alive_indices):
            if not keep_mask[row_idx]:
                if configs[cfg_idx].alive:
                    configs[cfg_idx].alive = False
                    eliminated_any = True
        return eliminated_any

    def _update_elite_archive(self, configs: List[ConfigState]) -> None:
        """
        Update the elite archive based on the current alive configurations.
        """
        if not self.scenario.use_elitist_restarts:
            return

        indices, agg_scores = self._compute_aggregated_scores(configs)
        if not indices:
            return

        n_alive = len(indices)
        if n_alive == 0:
            return

        k = max(1, int(math.ceil(self.scenario.elite_fraction * n_alive)))

        if self.task.maximize:
            order = np.argsort(-agg_scores)
        else:
            order = np.argsort(agg_scores)

        elite_entries: List[EliteEntry] = []
        for rank in range(min(k, n_alive)):
            row_idx = int(order[rank])
            cfg_idx = indices[row_idx]
            state = configs[cfg_idx]
            score = float(agg_scores[row_idx])
            elite_entries.append(EliteEntry(config=dict(state.config), score=score))

        all_elites = self._elite_archive + elite_entries

        if not all_elites:
            self._elite_archive = []
            return

        if self.task.maximize:
            all_elites.sort(key=lambda e: e.score, reverse=True)
        else:
            all_elites.sort(key=lambda e: e.score)

        self._elite_archive = all_elites[: self.scenario.max_elite_archive_size]

    def _num_alive(self, configs: List[ConfigState]) -> int:
        return sum(1 for c in configs if c.alive)

    def _rank_based_elimination(
        self,
        configs: List[ConfigState],
        scores: np.ndarray,
        alive_indices: List[int],
    ) -> bool:
        """
        Simple rank-based elimination used as a fallback or when there is not
        enough data for statistical tests.
        """
        n_alive = len(alive_indices)
        if n_alive <= 1:
            return False

        agg_scores = self._aggregate_rows(scores)
        order = np.argsort(-agg_scores if self.task.maximize else agg_scores)

        target_keep = max(
            self.scenario.min_survivors,
            int(math.ceil(n_alive * (1.0 - self.scenario.elimination_fraction))),
        )
        target_keep = max(1, min(target_keep, n_alive))

        keep_rows = set(int(idx) for idx in order[:target_keep])
        eliminated_any = False
        for row_idx, cfg_idx in enumerate(alive_indices):
            if row_idx not in keep_rows:
                if configs[cfg_idx].alive:
                    configs[cfg_idx].alive = False
                    eliminated_any = True
        return eliminated_any

    def _force_keep_top_k(
        self,
        configs: List[ConfigState],
        scores: np.ndarray,
        alive_indices: List[int],
        k: int,
    ) -> bool:
        """Ensure that at least k configs remain alive by keeping the k best."""
        n_alive = len(alive_indices)
        if n_alive <= k:
            return False

        agg_scores = self._aggregate_rows(scores)
        order = np.argsort(-agg_scores if self.task.maximize else agg_scores)
        keep_rows = set(int(idx) for idx in order[:k])

        eliminated_any = False
        for row_idx, cfg_idx in enumerate(alive_indices):
            new_alive = row_idx in keep_rows
            if configs[cfg_idx].alive != new_alive:
                configs[cfg_idx].alive = new_alive
                eliminated_any = True
        return eliminated_any

    def _make_neighbor_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new configuration by applying small perturbations to a base
        configuration (local search / neighborhood move).
        """
        cfg: Dict[str, Any] = dict(base_config)

        for name, spec in self.param_space.params.items():
            if name not in base_config:
                tmp = self.param_space.sample(self.rng)
                cfg[name] = tmp[name]
                continue

            current_value = base_config[name]

            if isinstance(spec, Real):
                low = spec.low
                high = spec.high
                range_width = high - low
                if range_width <= 0:
                    cfg[name] = float(current_value)
                    continue
                sigma = 0.1 * range_width
                new_val = float(current_value) + float(self.rng.normal(0.0, sigma))
                new_val = max(low, min(high, new_val))
                cfg[name] = new_val
            elif isinstance(spec, Int):
                low = spec.low
                high = spec.high
                range_width = max(1, high - low)
                step = max(1, int(round(0.1 * range_width)))
                delta = int(self.rng.integers(-step, step + 1))
                new_val = int(current_value) + delta
                new_val = max(low, min(high, new_val))
                cfg[name] = new_val
            elif isinstance(spec, Categorical):
                choices = list(spec.choices)
                if not choices:
                    cfg[name] = current_value
                    continue
                keep_prob = 0.7
                if current_value in choices and self.rng.random() < keep_prob:
                    cfg[name] = current_value
                else:
                    available = [c for c in choices if c != current_value] or choices
                    idx = int(self.rng.integers(0, len(available)))
                    cfg[name] = available[idx]
            else:
                cfg[name] = current_value

        return cfg

    def _refill_population(self, configs: List[ConfigState]) -> None:
        """
        If elitist restarts are enabled and the number of alive configurations
        is below the target population size, spawn new configurations to refill.
        """
        if not self.scenario.use_elitist_restarts:
            return

        target_pop = self.scenario.target_population_size or self.max_initial_configs
        alive_states = [c for c in configs if c.alive]
        n_alive = len(alive_states)

        if n_alive >= target_pop:
            return

        n_to_spawn = target_pop - n_alive
        if n_to_spawn <= 0:
            return

        n_neighbors = int(round(self.scenario.neighbor_fraction * n_to_spawn))
        n_neighbors = max(0, min(n_neighbors, n_to_spawn))
        n_fresh = n_to_spawn - n_neighbors

        elite_configs: List[Dict[str, Any]] = [e.config for e in self._elite_archive]
        if not elite_configs:
            elite_configs = [c.config for c in alive_states]

        for _ in range(n_neighbors):
            base_cfg = elite_configs[int(self.rng.integers(0, len(elite_configs)))]
            neighbor_cfg = self._make_neighbor_config(base_cfg)
            state = ConfigState(
                config_id=self._next_config_id,
                config=neighbor_cfg,
                alive=True,
            )
            self._next_config_id += 1
            configs.append(state)

        for _ in range(n_fresh):
            cfg = self.sampler.sample(self.rng)
            state = ConfigState(
                config_id=self._next_config_id,
                config=cfg,
                alive=True,
            )
            self._next_config_id += 1
            configs.append(state)

    def _aggregate_rows(self, scores: np.ndarray) -> np.ndarray:
        """Apply task aggregator row-wise to a score matrix."""
        return np.asarray([float(self.task.aggregator(row.tolist())) for row in scores], dtype=float)


def example_racing_usage(
    task: TuningTask, eval_fn: Callable[[Dict[str, Any], EvalContext], float]
) -> None:
    """
    Minimal usage example. Pass a TuningTask and evaluation function to run.
    """
    scenario = Scenario(
        max_experiments=500,
        min_survivors=2,
        elimination_fraction=0.5,
        instance_order_random=True,
        seed_order_random=True,
        start_instances=1,
        verbose=True,
        use_statistical_tests=True,
        alpha=0.05,
        min_blocks_before_elimination=3,
        use_adaptive_budget=True,
        initial_budget_per_run=5000,
        max_budget_per_run=20000,
        budget_growth_factor=2.0,
    )

    tuner = RacingTuner(
        task=task,
        scenario=scenario,
        seed=42,
        max_initial_configs=20,
    )

    best_config, history = tuner.run(eval_fn)
    print("Racing best config:", best_config)
    print("Number of evaluated configs:", len(history))


def example_model_based_racing_usage(
    task: TuningTask, eval_fn: Callable[[Dict[str, Any], EvalContext], float]
) -> None:
    """
    Usage example with a model-based sampler that learns from survivors.
    """
    scenario = Scenario(
        max_experiments=1000,
        min_survivors=2,
        elimination_fraction=0.5,
        instance_order_random=True,
        seed_order_random=True,
        start_instances=1,
        verbose=True,
        use_statistical_tests=True,
        alpha=0.05,
        min_blocks_before_elimination=3,
        use_adaptive_budget=True,
        initial_budget_per_run=5000,
        max_budget_per_run=20000,
        budget_growth_factor=2.0,
        use_elitist_restarts=True,
        target_population_size=20,
        elite_fraction=0.3,
        max_elite_archive_size=20,
        neighbor_fraction=0.5,
    )

    sampler = ModelBasedSampler(
        param_space=task.param_space,
        exploration_prob=0.2,
        min_samples_to_model=5,
    )

    tuner = RacingTuner(
        task=task,
        scenario=scenario,
        seed=42,
        max_initial_configs=20,
        sampler=sampler,
    )

    best_config, history = tuner.run(eval_fn)
    print("Model-based racing best config:", best_config)
    print("Number of evaluated configs:", len(history))


__all__ = ["ConfigState", "EliteEntry", "RacingTuner"]
