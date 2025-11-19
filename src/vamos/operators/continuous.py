from __future__ import annotations

import numpy as np


class ContinuousVariationWorkspace:
    """
    Simple buffer registry that hands out reusable NumPy arrays keyed by name.
    Operators can request scratch buffers instead of reallocating temporary
    arrays every generation.
    """

    def __init__(self):
        self._buffers: dict[str, np.ndarray] = {}

    def request(self, key: str, shape: tuple[int, ...], dtype) -> np.ndarray:
        dtype = np.dtype(dtype)
        buf = self._buffers.get(key)
        if buf is None or buf.shape != shape or buf.dtype != dtype:
            buf = np.empty(shape, dtype=dtype)
            self._buffers[key] = buf
        return buf


def _as_bound_vector(bound, length: int) -> np.ndarray:
    """
    Normalize bound inputs (scalar or vector) to a 1-D array of length `length`.
    """
    arr = np.asarray(bound, dtype=float)
    if arr.ndim == 0:
        return np.full(length, float(arr))
    if arr.shape[0] != length:
        raise ValueError(f"Expected bound with length {length}, got {arr.shape}")
    return arr.astype(float, copy=False)


def _repair_values(values, lower, upper, rng: np.random.Generator, strategy: str):
    strategy = (strategy or "clip").lower()
    if strategy == "random":
        mask_low = values < lower
        mask_high = values > upper
        mask = mask_low | mask_high
        if not np.any(mask):
            return values
        span = upper - lower
        rand = rng.random(mask.sum())
        repaired = lower[mask] + rand * span[mask]
        values = values.copy()
        values[mask] = repaired
        return values
    if strategy != "clip":
        raise ValueError(f"Unknown repair strategy '{strategy}'.")
    return np.clip(values, lower, upper)


def blx_alpha_crossover(
    X_parents: np.ndarray,
    prob: float,
    alpha: float,
    rng: np.random.Generator,
    xl,
    xu,
    *,
    repair: str = "clip",
    workspace: ContinuousVariationWorkspace | None = None,
    xl_vec: np.ndarray | None = None,
    xu_vec: np.ndarray | None = None,
) -> np.ndarray:
    """
    Batched BLX-alpha crossover with per-variable clipping to [xl, xu].
    """
    Np, D = X_parents.shape
    if Np == 0:
        return np.empty_like(X_parents)
    if Np % 2 != 0:
        raise ValueError("BLX-alpha crossover expects an even number of parents.")
    prob = float(np.clip(prob, 0.0, 1.0))
    alpha = float(alpha)
    offspring = X_parents.reshape(Np // 2, 2, D).copy()
    if prob <= 0.0:
        return offspring.reshape(Np, D)

    prob_buffer = None
    if workspace is not None:
        prob_buffer = workspace.request("blx_prob", (offspring.shape[0],), np.float64)
        rng.random(out=prob_buffer)
        mask = workspace.request("blx_mask", (offspring.shape[0],), np.bool_)
        np.less_equal(prob_buffer, prob, out=mask)
    else:
        mask = rng.random(offspring.shape[0]) <= prob
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return offspring.reshape(Np, D)

    if xl_vec is None:
        xl_vec = _as_bound_vector(xl, D)
    if xu_vec is None:
        xu_vec = _as_bound_vector(xu, D)
    parents = offspring[idx]
    p1 = parents[:, 0, :]
    p2 = parents[:, 1, :]
    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    span = hi - lo
    lower = lo - alpha * span
    upper = hi + alpha * span
    width = upper - lower
    rand_shape = lower.shape
    if workspace is not None:
        rand_1 = workspace.request("blx_rand_1", rand_shape, np.float64)
        rand_2 = workspace.request("blx_rand_2", rand_shape, np.float64)
        rng.random(out=rand_1)
        rng.random(out=rand_2)
    else:
        rand_1 = rng.random(rand_shape)
        rand_2 = rng.random(rand_shape)
    child1 = lower + rand_1 * width
    child2 = lower + rand_2 * width
    child1 = _repair_values(child1, xl_vec, xu_vec, rng, repair)
    child2 = _repair_values(child2, xl_vec, xu_vec, rng, repair)
    parents[:, 0, :] = child1
    parents[:, 1, :] = child2
    return offspring.reshape(Np, D)


def non_uniform_mutation(
    X: np.ndarray,
    prob: float,
    perturbation: float,
    rng: np.random.Generator,
    xl,
    xu,
    workspace: ContinuousVariationWorkspace | None = None,
    xl_vec: np.ndarray | None = None,
    xu_vec: np.ndarray | None = None,
) -> None:
    """
    Non-uniform mutation that shrinks perturbations via an exponential factor.
    Unlike the time-dependent textbook variant, `perturbation` directly
    controls how aggressively the noise decays toward the bounds.
    """
    N, D = X.shape
    if N == 0:
        return
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return
    perturbation = max(perturbation, 1e-8)

    if workspace is not None:
        prob_draws = workspace.request("nu_prob", (N, D), np.float64)
        rng.random(out=prob_draws)
        mask = workspace.request("nu_mask", (N, D), np.bool_)
        np.less_equal(prob_draws, prob, out=mask)
    else:
        mask = rng.random((N, D)) <= prob
    if not np.any(mask):
        return

    if xl_vec is None:
        xl_vec = _as_bound_vector(xl, D)
    if xu_vec is None:
        xu_vec = _as_bound_vector(xu, D)
    span = xu_vec - xl_vec

    if workspace is not None:
        rand = workspace.request("nu_rand", (N, D), np.float64)
        delta = workspace.request("nu_delta", (N, D), np.float64)
        rng.random(out=rand)
        np.power(rand, perturbation, out=rand)
        np.subtract(1.0, rand, out=delta)
        delta *= span
        direction_coin = workspace.request("nu_direction_coin", (N, D), np.float64)
        rng.random(out=direction_coin)
        direction_mask = workspace.request("nu_direction_mask", (N, D), np.bool_)
        np.less_equal(direction_coin, 0.5, out=direction_mask)
        delta[direction_mask] *= -1.0
    else:
        rand = rng.random((N, D))
        delta = (1.0 - rand ** perturbation) * span
        direction = np.where(rng.random((N, D)) <= 0.5, -1.0, 1.0)
        delta = direction * delta

    delta[~mask] = 0.0
    X += delta
    np.clip(X, xl_vec, xu_vec, out=X)
