from __future__ import annotations

from typing import Any
from collections.abc import Sequence

import numpy as np


def build_schedule(
    instances: Sequence[Any],
    seeds: Sequence[int],
    *,
    start_instances: int,
    instance_order_random: bool,
    seed_order_random: bool,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
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

    schedule: list[tuple[int, int]] = []

    # Interleave the initially-selected subset with the remaining instances.
    # This reduces early overfitting to `start_instances` while still giving
    # them slight priority.
    interleaved_instances: list[int] = []
    max_len = max(len(stage1_instances), len(remaining_instances))
    for i in range(max_len):
        if i < len(stage1_instances):
            interleaved_instances.append(stage1_instances[i])
        if i < len(remaining_instances):
            interleaved_instances.append(remaining_instances[i])

    for seed_idx in seed_indices:
        for inst_idx in interleaved_instances:
            schedule.append((inst_idx, seed_idx))

    return schedule


__all__ = ["build_schedule"]
