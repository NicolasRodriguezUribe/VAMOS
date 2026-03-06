from __future__ import annotations

from typing import Any

import numpy as np

_sbx_single_pair_numba = None


def _get_sbx_single_pair_numba() -> Any:
    """Lazily compile the Numba SBX kernel on first use."""
    global _sbx_single_pair_numba  # noqa: PLW0603
    if _sbx_single_pair_numba is not None:
        return _sbx_single_pair_numba
    try:
        from numba import njit

        @njit(cache=True)
        def _kernel(  # type: ignore[no-untyped-def]
            p1,
            p2,
            xl,
            xu,
            eta,
            prob_var,
            rand_var,
            rand_sbx,
            rand_swap,
        ):
            n = p1.shape[0]
            c1 = p1.copy()
            c2 = p2.copy()
            eps = 1.0e-14
            inv_eta = 1.0 / (eta + 1.0)
            neg_eta1 = -(eta + 1.0)
            for j in range(n):
                if rand_var[j] > prob_var:
                    continue
                y1 = min(p1[j], p2[j])
                y2 = max(p1[j], p2[j])
                diff = y2 - y1
                if diff <= eps:
                    continue
                r = rand_sbx[j]
                beta = 1.0 + 2.0 * (y1 - xl[j]) / diff
                if beta < eps:
                    beta = eps
                alpha = 2.0 - beta**neg_eta1
                if alpha < eps:
                    alpha = eps
                if r <= 1.0 / alpha:
                    betaq = (r * alpha) ** inv_eta
                else:
                    betaq = (1.0 / (2.0 - r * alpha)) ** inv_eta
                v1 = 0.5 * ((y1 + y2) - betaq * diff)

                beta = 1.0 + 2.0 * (xu[j] - y2) / diff
                if beta < eps:
                    beta = eps
                alpha = 2.0 - beta**neg_eta1
                if alpha < eps:
                    alpha = eps
                if r <= 1.0 / alpha:
                    betaq = (r * alpha) ** inv_eta
                else:
                    betaq = (1.0 / (2.0 - r * alpha)) ** inv_eta
                v2 = 0.5 * ((y1 + y2) + betaq * diff)

                if rand_swap[j] <= 0.5:
                    c1[j] = v2
                    c2[j] = v1
                else:
                    c1[j] = v1
                    c2[j] = v2
            return c1, c2

        _sbx_single_pair_numba = _kernel
        return _kernel
    except ImportError:
        return None


def _ensure_bounds(lower: np.ndarray, upper: np.ndarray, n_var: int) -> tuple[np.ndarray, np.ndarray]:
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    if lower_arr.ndim == 0 or (lower_arr.ndim == 1 and lower_arr.shape[0] == 1 and n_var > 1):
        lower_arr = np.full(n_var, float(lower_arr.reshape(-1)[0]))
    if upper_arr.ndim == 0 or (upper_arr.ndim == 1 and upper_arr.shape[0] == 1 and n_var > 1):
        upper_arr = np.full(n_var, float(upper_arr.reshape(-1)[0]))
    return lower_arr, upper_arr


def sbx_crossover_pairs(
    parents: np.ndarray,
    *,
    rng: np.random.Generator,
    lower: np.ndarray,
    upper: np.ndarray,
    prob_crossover: float = 0.9,
    eta: float = 10.0,
    prob_var: float = 0.5,
    inplace: bool = False,
) -> np.ndarray:
    offspring = parents if inplace else np.array(parents, copy=True, dtype=float)
    if offspring.ndim != 3 or offspring.shape[1] != 2:
        raise ValueError("SBX expects a mating tensor of shape (n_pairs, 2, n_var).")
    n_pairs, _, n_var = offspring.shape
    if n_pairs == 0:
        return offspring

    lower_arr, upper_arr = _ensure_bounds(lower, upper, n_var)

    if n_pairs == 1:
        kernel = _get_sbx_single_pair_numba()
        if kernel is not None:
            if rng.random() > prob_crossover:
                return offspring
            if prob_var < 1.0:
                rand_var = np.asarray(rng.random(n_var))
            else:
                rand_var = np.zeros(n_var, dtype=np.float64)
            rand_sbx = rng.random(n_var)
            rand_swap = rng.random(n_var)
            c1, c2 = kernel(
                offspring[0, 0, :],
                offspring[0, 1, :],
                lower_arr,
                upper_arr,
                float(eta),
                float(prob_var),
                rand_var,
                rand_sbx,
                rand_swap,
            )
            offspring[0, 0, :] = c1
            offspring[0, 1, :] = c2
            return offspring

    pair_mask = rng.random(n_pairs) <= prob_crossover
    idx = np.flatnonzero(pair_mask)
    if idx.size == 0:
        return offspring

    parent1 = offspring[idx, 0, :]
    parent2 = offspring[idx, 1, :]
    base1 = parent1.copy()
    base2 = parent2.copy()
    eps = 1.0e-14

    y1 = np.minimum(parent1, parent2)
    y2 = np.maximum(parent1, parent2)
    diff = y2 - y1
    valid = diff > eps
    if prob_var >= 1.0:
        active = valid
    else:
        active = valid & (rng.random(parent1.shape) <= prob_var)
    if not np.any(active):
        return offspring

    xl = lower_arr.reshape(1, -1)
    xu = upper_arr.reshape(1, -1)
    rand = rng.random(parent1.shape)
    betaq = np.empty_like(parent1)

    beta_valid = 1.0 + (2.0 * (y1 - xl) / diff.clip(min=eps))
    beta_valid = np.maximum(beta_valid, eps)
    alpha = 2.0 - np.power(beta_valid, -(eta + 1.0))
    alpha = np.maximum(alpha, eps)
    term = rand <= (1.0 / alpha)
    inv_eta = 1.0 / (eta + 1.0)
    betaq[term] = np.power(rand[term] * alpha[term], inv_eta)
    betaq[~term] = np.power(1.0 / (2.0 - rand[~term] * alpha[~term]), inv_eta)
    c1 = 0.5 * ((y1 + y2) - betaq * diff)

    beta_valid = 1.0 + (2.0 * (xu - y2) / diff.clip(min=eps))
    beta_valid = np.maximum(beta_valid, eps)
    alpha = 2.0 - np.power(beta_valid, -(eta + 1.0))
    alpha = np.maximum(alpha, eps)
    term = rand <= (1.0 / alpha)
    betaq[term] = np.power(rand[term] * alpha[term], inv_eta)
    betaq[~term] = np.power(1.0 / (2.0 - rand[~term] * alpha[~term]), inv_eta)
    c2 = 0.5 * ((y1 + y2) + betaq * diff)

    swap_mask = (rng.random(parent1.shape) <= 0.5) & active
    child1 = np.where(active, c1, base1)
    child2 = np.where(active, c2, base2)
    new_child1 = np.where(swap_mask, child2, child1)
    new_child2 = np.where(swap_mask, child1, child2)

    offspring[idx, 0, :] = new_child1
    offspring[idx, 1, :] = new_child2
    return offspring


def polynomial_mutation_population(
    offspring: np.ndarray,
    *,
    rng: np.random.Generator,
    lower: np.ndarray,
    upper: np.ndarray,
    prob_mutation: float,
    eta: float = 20.0,
    inplace: bool = False,
) -> np.ndarray:
    X = offspring if inplace else np.array(offspring, copy=True, dtype=float)
    if X.ndim != 2:
        raise ValueError("Polynomial mutation expects a population matrix of shape (n_ind, n_var).")
    n_ind, n_var = X.shape
    if n_ind == 0:
        return X

    lower_arr, upper_arr = _ensure_bounds(lower, upper, n_var)
    rnd_mask = rng.random(X.shape)
    rnd_delta = rng.random(X.shape)
    mask = rnd_mask <= prob_mutation
    if not np.any(mask):
        return X

    mut_pow = 1.0 / (eta + 1.0)
    rows, cols = np.nonzero(mask)
    for i, j in zip(rows, cols):
        y = X[i, j]
        yl = lower_arr[j]
        yu = upper_arr[j]
        if yu <= yl:
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
        X[i, j] = y + deltaq * (yu - yl)
    return X


__all__ = [
    "polynomial_mutation_population",
    "sbx_crossover_pairs",
]
