from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

import numpy as np
from numba import njit as _numba_njit

_F = TypeVar("_F", bound=Callable[..., object])


def njit(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:
    """Typed wrapper around numba.njit to keep mypy happy."""
    return cast(Callable[[_F], _F], _numba_njit(*args, **kwargs))


@njit(cache=True)
def _polynomial_mutation_numba_impl(
    X: np.ndarray,
    prob: float,
    eta: float,
    lower: np.ndarray,
    upper: np.ndarray,
    rnd_mask: np.ndarray,
    rnd_delta: np.ndarray,
) -> None:
    """Apply polynomial mutation in-place using caller-provided random draws."""
    n_ind, n_var = X.shape
    mut_pow = 1.0 / (eta + 1.0)

    for i in range(n_ind):
        for j in range(n_var):
            if rnd_mask[i, j] > prob:
                continue

            y = X[i, j]
            yl = lower[j]
            yu = upper[j]

            if yl >= yu:
                continue

            delta1 = (y - yl) / (yu - yl)
            delta2 = (yu - y) / (yu - yl)

            rnd = rnd_delta[i, j]

            if rnd <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (eta + 1.0))
                deltaq = val**mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (eta + 1.0))
                deltaq = 1.0 - val**mut_pow

            y = y + deltaq * (yu - yl)
            if y < yl:
                y = yl
            elif y > yu:
                y = yu
            X[i, j] = y


@njit(cache=True)
def polynomial_mutation_numba(
    X: np.ndarray,
    prob: float,
    eta: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    """Apply polynomial mutation in-place using Numba's internal RNG state."""
    rnd_mask = np.random.random(X.shape)
    rnd_delta = np.random.random(X.shape)
    _polynomial_mutation_numba_impl(X, prob, eta, lower, upper, rnd_mask, rnd_delta)


@njit(cache=True)
def tournament_winners_numba(
    ranks: np.ndarray,
    crowding: np.ndarray,
    candidates: np.ndarray,
    tie_break: np.ndarray,
) -> np.ndarray:
    """Resolve tournament winners for arbitrary pressure using caller-provided candidates."""
    n_parents, pressure = candidates.shape
    winners = np.empty(n_parents, dtype=np.int64)

    for i in range(n_parents):
        best = candidates[i, 0]
        best_rank = ranks[best]
        best_crowding = crowding[best]
        if np.isnan(best_crowding):
            best_crowding = -np.inf
        best_key = tie_break[i, 0]

        for j in range(1, pressure):
            cand = candidates[i, j]
            cand_rank = ranks[cand]
            cand_crowding = crowding[cand]
            if np.isnan(cand_crowding):
                cand_crowding = -np.inf
            cand_key = tie_break[i, j]

            if cand_rank < best_rank:
                best = cand
                best_rank = cand_rank
                best_crowding = cand_crowding
                best_key = cand_key
                continue
            if cand_rank > best_rank:
                continue
            if cand_crowding > best_crowding:
                best = cand
                best_crowding = cand_crowding
                best_key = cand_key
                continue
            if cand_crowding < best_crowding:
                continue
            if cand_key < best_key:
                best = cand
                best_key = cand_key

        winners[i] = best

    return winners


@njit(cache=True)
def sbx_crossover_numba(
    X_parents: np.ndarray,
    prob: float,
    eta: float,
    lower: np.ndarray,
    upper: np.ndarray,
    prob_var: float = 0.5,
) -> np.ndarray:
    """Apply SBX crossover to a flat parent matrix shaped ``(N, n_var)``."""
    n_parents_total, n_var = X_parents.shape
    n_parents = n_parents_total
    if n_parents % 2 != 0:
        n_parents -= 1

    offspring = np.empty_like(X_parents)

    # Process pairs
    for i in range(0, n_parents, 2):
        # pair i and i+1

        # Check probability for this pair
        if np.random.random() <= prob:
            for j in range(n_var):
                if np.random.random() > prob_var:
                    offspring[i, j] = X_parents[i, j]
                    offspring[i + 1, j] = X_parents[i + 1, j]
                    continue
                y1 = X_parents[i, j]
                y2 = X_parents[i + 1, j]

                yl = lower[j]
                yu = upper[j]

                if np.abs(y1 - y2) < 1.0e-14:
                    offspring[i, j] = y1
                    offspring[i + 1, j] = y2
                    continue

                if y1 < y2:
                    y1_val = y1
                    y2_val = y2
                else:
                    y1_val = y2
                    y2_val = y1

                # y1_val is min, y2_val is max

                beta = 1.0 + (2.0 * (y1_val - yl) / (y2_val - y1_val))
                alpha = 2.0 - beta ** -(eta + 1.0)
                rand = np.random.random()

                if rand <= (1.0 / alpha):
                    betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))

                c1 = 0.5 * ((y1_val + y2_val) - betaq * (y2_val - y1_val))

                beta = 1.0 + (2.0 * (yu - y2_val) / (y2_val - y1_val))
                alpha = 2.0 - beta ** -(eta + 1.0)

                if rand <= (1.0 / alpha):
                    betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))

                c2 = 0.5 * ((y1_val + y2_val) + betaq * (y2_val - y1_val))

                # Random swap is implied by the fact we sorted y1/y2?
                # Standard SBX often creates two children from y1, y2.
                # The order in output depends on implementation.
                # Standard pymoo/vamos:
                # child1 = 0.5*((y1+y2) - betaq*(y2-y1))
                # child2 = 0.5*((y1+y2) + betaq*(y2-y1))
                # And then typically we swap them with 0.5 prob per variable or per individual.
                # VAMOS real.py implementation does swap per variable (prob 0.5).

                if np.random.random() <= 0.5:
                    offspring[i, j] = c2
                    offspring[i + 1, j] = c1
                else:
                    offspring[i, j] = c1
                    offspring[i + 1, j] = c2
        else:
            # Copy parents
            for j in range(n_var):
                offspring[i, j] = X_parents[i, j]
                offspring[i + 1, j] = X_parents[i + 1, j]

    if n_parents_total % 2 != 0:
        for j in range(n_var):
            offspring[n_parents_total - 1, j] = X_parents[n_parents_total - 1, j]

    return offspring
