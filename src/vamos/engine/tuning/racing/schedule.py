from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np


def build_schedule(
    instances: Sequence[Any],
    seeds: Sequence[int],
    *,
    start_instances: int,
    instance_order_random: bool,
    seed_order_random: bool,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """
    Build a list of (instance_index, seed_index) pairs defining evaluation order.
    """
    n_instances = len(instances)
    n_seeds = len(seeds)
    if n_instances == 0 or n_seeds == 0:
        return []

    inst_indices = list(range(n_instances))
    seed_indices = list(range(n_seeds))

    if instance_order_random:
        rng.shuffle(inst_indices)
    if seed_order_random:
        rng.shuffle(seed_indices)

    k = max(1, min(start_instances, n_instances))
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


__all__ = ["build_schedule"]
