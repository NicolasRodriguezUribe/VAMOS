from __future__ import annotations

from typing import Any, Dict, List
import math

import numpy as np

from .param_space import ParamSpace, Real, Int, Categorical
from .sampler import Sampler
from .state import ConfigState, EliteEntry


def make_neighbor_config(base_config: Dict[str, Any], param_space: ParamSpace, rng: np.random.Generator) -> Dict[str, Any]:
    """
    Create a new configuration by applying small perturbations to a base configuration.
    """
    cfg: Dict[str, Any] = dict(base_config)

    for name, spec in param_space.params.items():
        if name not in base_config:
            tmp = param_space.sample(rng)
            cfg[name] = tmp[name]
            continue

        current_value = base_config[name]

        if isinstance(spec, Real):
            low = spec.low
            high = spec.high

            if spec.log:
                log_low = math.log(low)
                log_high = math.log(high)
                log_range = log_high - log_low
                sigma = 0.1 * log_range

                # Perturb in log-space
                cur_log = math.log(float(current_value))
                new_log = cur_log + float(rng.normal(0.0, sigma))
                new_val = math.exp(new_log)
            else:
                range_width = high - low
                if range_width <= 0:
                    cfg[name] = float(current_value)
                    continue
                sigma = 0.1 * range_width
                new_val = float(current_value) + float(rng.normal(0.0, sigma))

            new_val = max(low, min(high, new_val))
            cfg[name] = new_val
        elif isinstance(spec, Int):
            low = spec.low
            high = spec.high

            if spec.log:
                log_low = math.log(low)
                log_high = math.log(high)
                log_range = log_high - log_low
                sigma = max(0.1, 0.1 * log_range)  # ensure some variance

                cur_log = math.log(float(current_value))
                new_log = cur_log + float(rng.normal(0.0, sigma))
                new_val = int(round(math.exp(new_log)))
            else:
                range_width = max(1, high - low)
                step = max(1, int(round(0.1 * range_width)))
                delta = int(rng.integers(-step, step + 1))
                new_val = int(current_value) + delta

            new_val = max(low, min(high, new_val))
            cfg[name] = new_val
        elif isinstance(spec, Categorical):
            choices = list(spec.choices)
            if not choices:
                cfg[name] = current_value
                continue
            keep_prob = 0.7
            if current_value in choices and rng.random() < keep_prob:
                cfg[name] = current_value
            else:
                available = [c for c in choices if c != current_value] or choices
                idx = int(rng.integers(0, len(available)))
                cfg[name] = available[idx]
        else:
            cfg[name] = current_value

    return cfg


def refill_population(
    configs: List[ConfigState],
    *,
    scenario,
    param_space: ParamSpace,
    sampler: Sampler,
    elite_archive: List[EliteEntry],
    target_population_size: int,
    rng: np.random.Generator,
    next_config_id: int,
) -> int:
    """
    Spawn new configurations when elitist restarts are enabled and alive count is below target.
    Returns the updated next_config_id.
    """
    alive_states = [c for c in configs if c.alive]
    n_alive = len(alive_states)

    if n_alive >= target_population_size:
        return next_config_id

    n_to_spawn = target_population_size - n_alive
    if n_to_spawn <= 0:
        return next_config_id

    n_neighbors = int(round(scenario.neighbor_fraction * n_to_spawn))
    n_neighbors = max(0, min(n_neighbors, n_to_spawn))
    n_fresh = n_to_spawn - n_neighbors

    elite_configs: List[Dict[str, Any]] = [e.config for e in elite_archive]
    if not elite_configs:
        elite_configs = [c.config for c in alive_states]

    for _ in range(n_neighbors):
        base_cfg = elite_configs[int(rng.integers(0, len(elite_configs)))]
        neighbor_cfg = make_neighbor_config(base_cfg, param_space, rng)
        state = ConfigState(
            config_id=next_config_id,
            config=neighbor_cfg,
            alive=True,
        )
        next_config_id += 1
        configs.append(state)

    for _ in range(n_fresh):
        cfg = sampler.sample(rng)
        state = ConfigState(
            config_id=next_config_id,
            config=cfg,
            alive=True,
        )
        next_config_id += 1
        configs.append(state)

    return next_config_id


__all__ = ["make_neighbor_config", "refill_population"]
