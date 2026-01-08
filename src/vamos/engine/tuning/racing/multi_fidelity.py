from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from joblib import Parallel, delayed

from .state import ConfigState
from .tuning_task import EvalContext
from .random_search_tuner import TrialResult

if TYPE_CHECKING:
    from .core import RacingTuner


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _eval_worker_warmstart(
    eval_fn: Callable[[Dict[str, Any], EvalContext], Any],
    config: Dict[str, Any],
    ctx: EvalContext,
    warm_start: bool,
) -> Any:
    result = eval_fn(config, ctx)
    return result


def _run_multi_fidelity(
    tuner: "RacingTuner",
    eval_fn: Callable[[Dict[str, Any], EvalContext], float],
    verbose: Optional[bool] = None,
) -> Tuple[Dict[str, Any], List[TrialResult]]:
    verbose_flag = tuner.scenario.verbose if verbose is None else verbose
    fidelity_levels = list(tuner.scenario.fidelity_levels)
    promotion_ratio = tuner.scenario.fidelity_promotion_ratio
    min_configs = tuner.scenario.fidelity_min_configs

    configs: List[ConfigState] = tuner._sample_initial_configs()
    num_experiments = 0

    if verbose_flag:
        _logger().info(
            "[multi-fidelity] Starting with %d configs, %d fidelity levels: %s",
            len(configs),
            len(fidelity_levels),
            fidelity_levels,
        )

    for fidelity_idx, budget in enumerate(fidelity_levels):
        alive_count = tuner._num_alive(configs)
        if alive_count == 0:
            if verbose_flag:
                _logger().info("[multi-fidelity] No configs alive, stopping.")
            break

        if num_experiments + alive_count > tuner.scenario.max_experiments:
            if verbose_flag:
                _logger().info("[multi-fidelity] Experiment budget exhausted.")
            break

        if verbose_flag:
            _logger().info(
                "[multi-fidelity] Fidelity %d/%d (budget=%d): evaluating %d configs",
                fidelity_idx + 1,
                len(fidelity_levels),
                budget,
                alive_count,
            )

        inst_idx = fidelity_idx % len(tuner.instances)
        seed_idx = fidelity_idx % len(tuner.seeds)

        _run_stage_with_budget(tuner, configs, inst_idx, seed_idx, eval_fn, budget, fidelity_idx)
        num_experiments += alive_count

        scored_configs: List[Tuple[int, float]] = []
        for idx, state in enumerate(configs):
            if not state.alive:
                continue
            if not state.scores:
                continue
            agg = float(tuner.task.aggregator(state.scores))
            scored_configs.append((idx, agg))

        if tuner.task.maximize:
            scored_configs.sort(key=lambda x: x[1], reverse=True)
        else:
            scored_configs.sort(key=lambda x: x[1])

        is_final_level = fidelity_idx == len(fidelity_levels) - 1
        if is_final_level:
            n_keep = max(tuner.scenario.min_survivors, min_configs)
        else:
            n_keep = max(int(len(scored_configs) * promotion_ratio), min_configs)

        survivors = set(idx for idx, _ in scored_configs[:n_keep])
        eliminated = 0
        for idx, state in enumerate(configs):
            if state.alive and idx not in survivors:
                state.alive = False
                eliminated += 1

        if verbose_flag and eliminated > 0:
            _logger().info(
                "[multi-fidelity] Eliminated %d configs, %d survivors promoted",
                eliminated,
                len(survivors),
            )

    best_state, history = tuner._finalize_results(configs)

    if best_state is None:
        raise RuntimeError("Multi-fidelity tuning finished without a valid configuration.")

    if verbose_flag and best_state.score is not None:
        _logger().info(
            "[multi-fidelity] Best score=%.6f after %d fidelity levels.",
            best_state.score,
            len(fidelity_levels),
        )

    return best_state.config, history


def _run_stage_with_budget(
    tuner: "RacingTuner",
    configs: List[ConfigState],
    inst_idx: int,
    seed_idx: int,
    eval_fn: Callable[[Dict[str, Any], EvalContext], float],
    budget: int,
    fidelity_level: int = 0,
) -> None:
    instance = tuner.instances[inst_idx]
    seed = tuner.seeds[seed_idx]
    warm_start = tuner.scenario.fidelity_warm_start

    tasks = []
    indices = []
    for idx, state in enumerate(configs):
        if not state.alive:
            continue

        ctx = EvalContext(
            instance=instance,
            seed=seed,
            budget=budget,
            fidelity_level=fidelity_level,
            previous_budget=state.last_budget if warm_start else None,
            checkpoint=state.checkpoint if warm_start else None,
        )
        tasks.append((state.config, ctx, idx))
        indices.append(idx)

    if not tasks:
        return

    if tuner.scenario.n_jobs == 1:
        for cfg, ctx, idx in tasks:
            result = eval_fn(cfg, ctx)

            if warm_start and isinstance(result, tuple) and len(result) == 2:
                score, checkpoint = result
                configs[idx].checkpoint = checkpoint
            else:
                score = float(result) if not isinstance(result, tuple) else float(result[0])

            configs[idx].scores.append(float(score))
            configs[idx].last_budget = budget
    else:
        results = Parallel(n_jobs=tuner.scenario.n_jobs)(
            delayed(_eval_worker_warmstart)(eval_fn, cfg, ctx, warm_start) for cfg, ctx, _ in tasks
        )
        for i, result in enumerate(results):
            idx = tasks[i][2]
            if warm_start and isinstance(result, tuple) and len(result) == 2:
                score, checkpoint = result
                configs[idx].checkpoint = checkpoint
            else:
                score = float(result) if not isinstance(result, tuple) else float(result[0])

            configs[idx].scores.append(float(score))
            configs[idx].last_budget = budget


__all__ = ["_run_multi_fidelity", "_run_stage_with_budget"]
