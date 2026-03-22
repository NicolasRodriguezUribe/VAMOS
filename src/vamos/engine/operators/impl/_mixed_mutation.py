from __future__ import annotations

from typing import Any

import numpy as np

from ._mixed_spec import (
    CUSTOM_MUTATION_KEYS,
    extract_index_array,
    has_customized_segment_settings,
    resolve_choice,
    resolve_perm_mutation,
    resolve_probability,
    validate_mixed_spec,
    validate_segment_bounds,
)
from .integer import creep_mutation, integer_polynomial_mutation, random_reset_mutation


def _mixed_mutation_default(X: np.ndarray, prob: float, spec: dict[str, Any], rng: np.random.Generator) -> None:
    if X.size == 0:
        return
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return

    validate_mixed_spec(spec, X.shape[1])
    perm_idx = extract_index_array(spec, "perm_idx")
    real_idx = extract_index_array(spec, "real_idx")
    int_idx = extract_index_array(spec, "int_idx")
    cat_idx = extract_index_array(spec, "cat_idx")
    real_lower = np.asarray(spec.get("real_lower") if spec.get("real_lower") is not None else [], dtype=float)
    real_upper = np.asarray(spec.get("real_upper") if spec.get("real_upper") is not None else [], dtype=float)
    int_lower = np.asarray(spec.get("int_lower") if spec.get("int_lower") is not None else [], dtype=int)
    int_upper = np.asarray(spec.get("int_upper") if spec.get("int_upper") is not None else [], dtype=int)
    cat_cardinality = np.asarray(spec.get("cat_cardinality") if spec.get("cat_cardinality") is not None else [], dtype=int)
    perm_mutation = resolve_perm_mutation(spec) if perm_idx.size else None

    if perm_idx.size:
        perm_view = X[:, perm_idx].astype(np.int32, copy=True)
        assert perm_mutation is not None
        perm_mutation(perm_view, prob, rng)
        X[:, perm_idx] = perm_view

    if real_idx.size:
        span = np.maximum(real_upper - real_lower, 1e-6)
        noise = rng.normal(scale=0.1 * span, size=(X.shape[0], real_idx.size))
        mask = rng.random((X.shape[0], real_idx.size)) <= prob
        proposed = np.clip(X[:, real_idx] + noise, real_lower, real_upper)
        X[:, real_idx] = np.where(mask, proposed, X[:, real_idx])

    if int_idx.size:
        mask = rng.random((X.shape[0], int_idx.size)) <= prob
        if np.any(mask):
            rand_vals = rng.integers(int_lower, int_upper + 1, size=(X.shape[0], int_idx.size), dtype=np.int32)
            X_int = X[:, int_idx]
            X_int[mask] = rand_vals[mask]
            X[:, int_idx] = X_int

    if cat_idx.size:
        mask = rng.random((X.shape[0], cat_idx.size)) <= prob
        if np.any(mask):
            cats = np.empty((X.shape[0], cat_idx.size), dtype=np.int32)
            for j in range(cat_idx.size):
                cats[:, j] = rng.integers(0, int(cat_cardinality[j]), size=X.shape[0], dtype=np.int32)
            X_cat = X[:, cat_idx]
            X_cat[mask] = cats[mask]
            X[:, cat_idx] = X_cat


def mixed_mutation(X: np.ndarray, prob: float, spec: dict[str, Any], rng: np.random.Generator) -> None:
    """
    Mixed mutation: permutation mutation for perm_idx, Gaussian perturb for real,
    random reset for int/cat.
    """
    if not has_customized_segment_settings(spec, CUSTOM_MUTATION_KEYS):
        _mixed_mutation_default(X, prob, spec, rng)
        return
    if X.size == 0:
        return

    validate_mixed_spec(spec, X.shape[1])
    perm_idx = extract_index_array(spec, "perm_idx")
    real_idx = extract_index_array(spec, "real_idx")
    int_idx = extract_index_array(spec, "int_idx")
    cat_idx = extract_index_array(spec, "cat_idx")
    real_lower = np.asarray(spec.get("real_lower") if spec.get("real_lower") is not None else [], dtype=float)
    real_upper = np.asarray(spec.get("real_upper") if spec.get("real_upper") is not None else [], dtype=float)
    int_lower = np.asarray(spec.get("int_lower") if spec.get("int_lower") is not None else [], dtype=int)
    int_upper = np.asarray(spec.get("int_upper") if spec.get("int_upper") is not None else [], dtype=int)
    cat_cardinality = np.asarray(spec.get("cat_cardinality") if spec.get("cat_cardinality") is not None else [], dtype=int)
    validate_segment_bounds(index_name="real_idx", idx=real_idx, lower_name="real_lower", lower=real_lower, upper_name="real_upper", upper=real_upper)
    validate_segment_bounds(index_name="int_idx", idx=int_idx, lower_name="int_lower", lower=int_lower, upper_name="int_upper", upper=int_upper)
    if cat_idx.size and cat_cardinality.shape[0] != cat_idx.size:
        raise ValueError("cat_cardinality length must match cat_idx size.")

    perm_prob = resolve_probability(spec, "perm_mutation_prob", prob)
    real_prob = resolve_probability(spec, "real_mutation_prob", prob)
    int_prob = resolve_probability(spec, "int_mutation_prob", prob)
    cat_prob = resolve_probability(spec, "cat_mutation_prob", prob)
    real_method = resolve_choice(spec, "real_mutation", "gaussian", allowed={"gaussian", "reset", "uniform_reset", "random_reset", "polynomial"})
    int_method = resolve_choice(spec, "int_mutation", "reset", allowed={"reset", "random_reset", "creep", "polynomial"})
    cat_method = resolve_choice(spec, "cat_mutation", "reset", allowed={"reset", "uniform_reset", "random_reset"})

    if perm_idx.size and perm_prob > 0.0:
        perm_view = X[:, perm_idx].astype(np.int32, copy=True)
        resolve_perm_mutation(spec)(perm_view, perm_prob, rng)
        X[:, perm_idx] = perm_view

    if real_idx.size and real_prob > 0.0:
        X_real = np.asarray(X[:, real_idx], dtype=float)
        if real_method == "gaussian":
            sigma_raw = spec.get("real_mutation_sigma")
            if sigma_raw is None:
                sigma = float(spec.get("real_mutation_sigma_factor", 0.1)) * np.maximum(real_upper - real_lower, 1.0e-6)
            else:
                sigma_arr = np.asarray(sigma_raw, dtype=float)
                if sigma_arr.ndim > 0 and sigma_arr.shape[0] != real_idx.size:
                    raise ValueError("real_mutation_sigma length must match real_idx size.")
                sigma = float(sigma_arr) if sigma_arr.ndim == 0 else sigma_arr
            mask = rng.random((X.shape[0], real_idx.size)) <= real_prob
            proposed = np.clip(X_real + rng.normal(scale=sigma, size=(X.shape[0], real_idx.size)), real_lower, real_upper)
            X_real = np.where(mask, proposed, X_real)
        elif real_method in {"reset", "uniform_reset", "random_reset"}:
            mask = rng.random((X.shape[0], real_idx.size)) <= real_prob
            resampled = rng.uniform(real_lower, real_upper, size=(X.shape[0], real_idx.size))
            X_real[mask] = resampled[mask]
        elif real_method == "polynomial":
            eta = float(spec.get("real_mutation_eta", 20.0))
            if eta <= 0.0:
                raise ValueError("real_mutation_eta must be > 0.")
            rnd_mask = rng.random((X.shape[0], real_idx.size))
            rnd_delta = rng.random((X.shape[0], real_idx.size))
            for i, j in zip(*np.nonzero(rnd_mask <= real_prob)):
                y = float(X_real[i, j])
                yl = float(real_lower[j])
                yu = float(real_upper[j])
                if yu <= yl:
                    continue
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                rnd = float(rnd_delta[i, j])
                mut_pow = 1.0 / (eta + 1.0)
                if rnd <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (eta + 1.0))
                    deltaq = val**mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val**mut_pow
                X_real[i, j] = min(max(y + deltaq * (yu - yl), yl), yu)
        else:
            raise ValueError(f"Unsupported real_mutation '{real_method}'.")
        X[:, real_idx] = X_real

    if int_idx.size and int_prob > 0.0:
        X_int = np.rint(X[:, int_idx]).astype(np.int32, copy=True)
        if int_method in {"reset", "random_reset"}:
            random_reset_mutation(X_int, int_prob, int_lower, int_upper, rng)
        elif int_method == "creep":
            step = int(spec.get("int_mutation_step", 1))
            if step <= 0:
                raise ValueError("int_mutation_step must be >= 1.")
            creep_mutation(X_int, int_prob, step, int_lower, int_upper, rng)
        elif int_method == "polynomial":
            eta = float(spec.get("int_mutation_eta", 20.0))
            if eta <= 0.0:
                raise ValueError("int_mutation_eta must be > 0.")
            integer_polynomial_mutation(X_int, int_prob, eta, int_lower, int_upper, rng)
        else:
            raise ValueError(f"Unsupported int_mutation '{int_method}'.")
        X[:, int_idx] = X_int

    if cat_idx.size and cat_prob > 0.0:
        if cat_method not in {"reset", "uniform_reset", "random_reset"}:
            raise ValueError(f"Unsupported cat_mutation '{cat_method}'.")
        mask = rng.random((X.shape[0], cat_idx.size)) <= cat_prob
        if np.any(mask):
            cats = np.empty((X.shape[0], cat_idx.size), dtype=np.int32)
            for cat_pos in range(int(cat_idx.size)):
                cats[:, cat_pos] = rng.integers(0, int(cat_cardinality[cat_pos]), size=X.shape[0], dtype=np.int32)
            X_cat = np.rint(X[:, cat_idx]).astype(np.int32, copy=True)
            X_cat[mask] = cats[mask]
            X[:, cat_idx] = X_cat


__all__ = ["mixed_mutation"]
