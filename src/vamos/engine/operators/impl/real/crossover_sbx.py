"""SBX-family real-valued crossover operators."""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.kernel.operator_primitives import sbx_crossover_pairs

from ._crossover_base import Crossover
from .utils import ArrayLike, VariationWorkspace, _ensure_bounds

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


class SBXCrossover(Crossover):
    """Simulated Binary Crossover (SBX) operator."""

    def __init__(
        self,
        prob_crossover: float = 0.9,
        eta: float = 10.0,
        prob_var: float = 0.5,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        workspace: VariationWorkspace | None = None,
        allow_inplace: bool = False,
    ) -> None:
        self.prob = float(prob_crossover)
        self.eta = float(eta)
        self.prob_var = float(prob_var)
        self.lower, self.upper = _ensure_bounds(lower, upper)
        self.workspace = workspace
        self.allow_inplace = bool(allow_inplace)

    def _rand(self, key: str, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        if self.workspace is None:
            return rng.random(shape)
        buf = self.workspace.request(key, shape, np.float64)
        rng.random(out=buf)
        return buf

    def _mask_pairs(self, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
        if self.workspace is None:
            return rng.random(n_pairs) <= self.prob
        probs = self.workspace.request("sbx_prob", (n_pairs,), np.float64)
        rng.random(out=probs)
        mask = self.workspace.request("sbx_mask", (n_pairs,), np.bool_)
        np.less_equal(probs, self.prob, out=mask)
        return mask

    def _buffer(self, key: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        if self.workspace is None:
            return np.empty(shape, dtype=dtype)
        return self.workspace.request(key, shape, dtype)

    def __call__(self, parents: ArrayLike, rng: np.random.Generator) -> ArrayLike:
        parents_arr = self._as_matings(parents, copy=False, name="parents")
        offspring = parents_arr if self.allow_inplace else parents_arr.copy()
        n_pairs, _, _ = offspring.shape
        if n_pairs == 0:
            return offspring
        self._check_bounds_match(offspring[:, 0, :], self.lower)
        if self.workspace is None:
            return sbx_crossover_pairs(
                offspring,
                rng=rng,
                lower=self.lower,
                upper=self.upper,
                prob_crossover=self.prob,
                eta=self.eta,
                prob_var=self.prob_var,
                inplace=True,
            )

        if n_pairs == 1:
            kernel = _get_sbx_single_pair_numba()
            if kernel is not None:
                if rng.random() > self.prob:
                    return offspring
                n_var = offspring.shape[2]
                rand_var = np.asarray(rng.random(n_var)) if self.prob_var < 1.0 else np.zeros(n_var, dtype=np.float64)
                rand_sbx = rng.random(n_var)
                rand_swap = rng.random(n_var)
                c1, c2 = kernel(
                    offspring[0, 0, :], offspring[0, 1, :], self.lower, self.upper, self.eta, self.prob_var, rand_var, rand_sbx, rand_swap
                )
                offspring[0, 0, :] = c1
                offspring[0, 1, :] = c2
                return offspring

        pair_mask = self._mask_pairs(n_pairs, rng)
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
        if self.prob_var >= 1.0:
            active = valid
        else:
            var_rand = self._rand("sbx_var", parent1.shape, rng)
            if self.workspace is None:
                var_mask = var_rand <= self.prob_var
            else:
                var_mask = self.workspace.request("sbx_var_mask", parent1.shape, np.bool_)
                np.less_equal(var_rand, self.prob_var, out=var_mask)
            active = valid & var_mask
        if not np.any(active):
            return offspring

        xl = self.lower.reshape(1, -1)
        xu = self.upper.reshape(1, -1)
        rand = self._rand("sbx_rand", parent1.shape, rng)
        betaq = self._buffer("sbx_betaq", parent1.shape, parent1.dtype)

        beta_valid = np.maximum(1.0 + (2.0 * (y1 - xl) / diff.clip(min=eps)), eps)
        alpha = np.maximum(2.0 - np.power(beta_valid, -(self.eta + 1.0)), eps)
        term = rand <= (1.0 / alpha)
        inv_eta = 1.0 / (self.eta + 1.0)
        betaq[term] = np.power(rand[term] * alpha[term], inv_eta)
        betaq[~term] = np.power(1.0 / (2.0 - rand[~term] * alpha[~term]), inv_eta)
        c1 = 0.5 * ((y1 + y2) - betaq * diff)

        beta_valid = np.maximum(1.0 + (2.0 * (xu - y2) / diff.clip(min=eps)), eps)
        alpha = np.maximum(2.0 - np.power(beta_valid, -(self.eta + 1.0)), eps)
        term = rand <= (1.0 / alpha)
        betaq[term] = np.power(rand[term] * alpha[term], inv_eta)
        betaq[~term] = np.power(1.0 / (2.0 - rand[~term] * alpha[~term]), inv_eta)
        c2 = 0.5 * ((y1 + y2) + betaq * diff)

        swap = self._rand("sbx_swap", parent1.shape, rng)
        if self.workspace is None:
            swap_mask = swap <= 0.5
        else:
            swap_mask = self.workspace.request("sbx_swap_mask", parent1.shape, np.bool_)
            np.less_equal(swap, 0.5, out=swap_mask)
        child1 = np.where(active, c1, base1)
        child2 = np.where(active, c2, base2)
        swap_mask = swap_mask & active
        offspring[idx, 0, :] = np.where(swap_mask, child2, child1)
        offspring[idx, 1, :] = np.where(swap_mask, child1, child2)
        return offspring


__all__ = ["SBXCrossover"]
