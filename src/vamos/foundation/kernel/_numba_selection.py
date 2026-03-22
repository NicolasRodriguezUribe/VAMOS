from __future__ import annotations

import numpy as np

from .numba_ops import njit


@njit(cache=True)
def binary_tournament_winners_numba(
    ranks: np.ndarray,
    crowding: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    tie_break: np.ndarray,
) -> np.ndarray:
    n_parents = first.shape[0]
    winners = np.empty(n_parents, dtype=np.int64)

    for i in range(n_parents):
        left = first[i]
        right = second[i]
        left_rank = ranks[left]
        right_rank = ranks[right]

        if left_rank < right_rank:
            winners[i] = left
            continue
        if right_rank < left_rank:
            winners[i] = right
            continue

        left_crowding = crowding[left]
        right_crowding = crowding[right]

        if np.isnan(left_crowding):
            left_crowding = -np.inf
        if np.isnan(right_crowding):
            right_crowding = -np.inf

        if left_crowding > right_crowding:
            winners[i] = left
            continue
        if right_crowding > left_crowding:
            winners[i] = right
            continue

        winners[i] = left if tie_break[i] == 0 else right

    return winners


__all__ = ["binary_tournament_winners_numba"]
