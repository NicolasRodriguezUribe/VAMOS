from __future__ import annotations

from typing import Any

import numpy as np

from .integer import (
    creep_mutation,
    integer_polynomial_mutation,
    integer_sbx_crossover,
    random_reset_mutation,
)
from .permutation import (
    alternating_edges_crossover,
    cycle_crossover,
    displacement_mutation,
    edge_recombination_crossover,
    insert_mutation,
    inversion_mutation,
    order_crossover,
    pmx_crossover,
    position_based_crossover,
    scramble_mutation,
    swap_mutation,
    two_opt_mutation,
)

_PERM_CROSSOVER = {
    "ox": order_crossover,
    "order": order_crossover,
    "oxd": order_crossover,
    "pmx": pmx_crossover,
    "cycle": cycle_crossover,
    "cx": cycle_crossover,
    "position": position_based_crossover,
    "position_based": position_based_crossover,
    "pos": position_based_crossover,
    "edge": edge_recombination_crossover,
    "edge_recombination": edge_recombination_crossover,
    "erx": edge_recombination_crossover,
    "aex": alternating_edges_crossover,
    "alternating_edges": alternating_edges_crossover,
}

_PERM_MUTATION = {
    "swap": swap_mutation,
    "insert": insert_mutation,
    "scramble": scramble_mutation,
    "inversion": inversion_mutation,
    "displacement": displacement_mutation,
    "two_opt": two_opt_mutation,
}


_CUSTOM_CROSSOVER_KEYS: set[str] = {
    "perm_crossover_prob",
    "real_crossover_prob",
    "int_crossover_prob",
    "cat_crossover_prob",
    "real_crossover",
    "int_crossover",
    "cat_crossover",
    "int_crossover_eta",
}

_CUSTOM_MUTATION_KEYS: set[str] = {
    "perm_mutation_prob",
    "real_mutation_prob",
    "int_mutation_prob",
    "cat_mutation_prob",
    "real_mutation",
    "int_mutation",
    "cat_mutation",
    "real_mutation_sigma",
    "real_mutation_sigma_factor",
    "real_mutation_eta",
    "int_mutation_step",
    "int_mutation_eta",
}


def _extract_index_array(spec: dict[str, Any], key: str) -> np.ndarray:
    raw = spec.get(key)
    if raw is None:
        return np.asarray([], dtype=int)
    return np.asarray(raw, dtype=int)


def _validate_mixed_spec(spec: dict[str, Any], n_var: int) -> None:
    indices = {
        "perm_idx": _extract_index_array(spec, "perm_idx"),
        "real_idx": _extract_index_array(spec, "real_idx"),
        "int_idx": _extract_index_array(spec, "int_idx"),
        "cat_idx": _extract_index_array(spec, "cat_idx"),
    }
    if not any(idx.size for idx in indices.values()):
        return
    all_idx = np.concatenate([idx for idx in indices.values() if idx.size])
    if np.any(all_idx < 0) or np.any(all_idx >= n_var):
        raise ValueError("mixed_spec indices must be within [0, n_var).")
    unique = np.unique(all_idx)
    if unique.size != all_idx.size:
        raise ValueError("mixed_spec indices must be disjoint across segments.")


def _resolve_perm_crossover(spec: dict[str, Any]) -> Any:
    method = str(spec.get("perm_crossover", "ox")).lower()
    try:
        return _PERM_CROSSOVER[method]
    except KeyError as exc:
        available = ", ".join(sorted(_PERM_CROSSOVER))
        raise ValueError(f"Unknown perm_crossover '{method}'. Available: {available}") from exc


def _resolve_perm_mutation(spec: dict[str, Any]) -> Any:
    method = str(spec.get("perm_mutation", "swap")).lower()
    try:
        return _PERM_MUTATION[method]
    except KeyError as exc:
        available = ", ".join(sorted(_PERM_MUTATION))
        raise ValueError(f"Unknown perm_mutation '{method}'. Available: {available}") from exc


def _resolve_probability(spec: dict[str, Any], key: str, fallback: float) -> float:
    raw = spec.get(key, fallback)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{key}' must be a real number in [0,1].") from exc
    return float(np.clip(value, 0.0, 1.0))


def _resolve_choice(
    spec: dict[str, Any],
    key: str,
    default: str,
    *,
    allowed: set[str],
) -> str:
    raw = str(spec.get(key, default)).strip().lower()
    if raw not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"Unknown {key} '{raw}'. Available: {options}")
    return raw


def _has_customized_segment_settings(spec: dict[str, Any], keys: set[str]) -> bool:
    return any(key in spec for key in keys)


def _validate_segment_bounds(
    *,
    index_name: str,
    idx: np.ndarray,
    lower_name: str,
    lower: np.ndarray,
    upper_name: str,
    upper: np.ndarray,
) -> None:
    if idx.size == 0:
        return
    if lower.shape[0] != idx.size or upper.shape[0] != idx.size:
        raise ValueError(f"{lower_name}/{upper_name} lengths must match {index_name} size.")


def _mixed_crossover_legacy(
    X_parents: np.ndarray,
    prob: float,
    spec: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    Np, D = X_parents.shape
    if Np == 0:
        return np.empty_like(X_parents)
    n_original = Np
    # Handle odd parent count by duplicating the last parent
    if Np % 2 != 0:
        X_parents = np.vstack([X_parents, X_parents[-1:]])
        Np += 1
    pairs = X_parents.reshape(Np // 2, 2, D).copy()
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        offspring = pairs.reshape(Np, D)
        if n_original % 2 != 0:
            offspring = offspring[:n_original]
        return offspring

    _validate_mixed_spec(spec, D)
    perm_idx = _extract_index_array(spec, "perm_idx")
    real_idx = _extract_index_array(spec, "real_idx")
    int_idx = _extract_index_array(spec, "int_idx")
    cat_idx = _extract_index_array(spec, "cat_idx")
    perm_crossover = _resolve_perm_crossover(spec) if perm_idx.size else None

    active = rng.random(pairs.shape[0]) <= prob
    act_idx = np.flatnonzero(active)
    if act_idx.size == 0:
        offspring = pairs.reshape(Np, D)
        if n_original % 2 != 0:
            offspring = offspring[:n_original]
        return offspring

    for row in act_idx:
        p1, p2 = pairs[row, 0], pairs[row, 1]
        child1 = p1.copy()
        child2 = p2.copy()
        if perm_idx.size:
            parents_perm = np.stack([p1[perm_idx], p2[perm_idx]], axis=0).astype(np.int32, copy=True)
            assert perm_crossover is not None
            perm_children = perm_crossover(parents_perm, 1.0, rng)
            child1[perm_idx] = perm_children[0]
            child2[perm_idx] = perm_children[1]
        if real_idx.size:
            mean_vals = 0.5 * (p1[real_idx] + p2[real_idx])
            child1[real_idx] = mean_vals
            child2[real_idx] = mean_vals
        if int_idx.size or cat_idx.size:
            swap_positions = np.concatenate([int_idx, cat_idx])
            if swap_positions.size:
                mask = rng.random(swap_positions.size) < 0.5
                swap_cols = swap_positions[mask]
                child1[swap_cols] = p2[swap_cols]
                child2[swap_cols] = p1[swap_cols]
        pairs[row, 0], pairs[row, 1] = child1, child2

    offspring = pairs.reshape(Np, D)
    if n_original % 2 != 0:
        offspring = offspring[:n_original]
    return offspring


def _mixed_mutation_legacy(
    X: np.ndarray,
    prob: float,
    spec: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    if X.size == 0:
        return
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return

    _validate_mixed_spec(spec, X.shape[1])
    perm_idx = _extract_index_array(spec, "perm_idx")
    real_idx = _extract_index_array(spec, "real_idx")
    int_idx = _extract_index_array(spec, "int_idx")
    cat_idx = _extract_index_array(spec, "cat_idx")
    real_lower = np.asarray(spec.get("real_lower") if spec.get("real_lower") is not None else [], dtype=float)
    real_upper = np.asarray(spec.get("real_upper") if spec.get("real_upper") is not None else [], dtype=float)
    int_lower = np.asarray(spec.get("int_lower") if spec.get("int_lower") is not None else [], dtype=int)
    int_upper = np.asarray(spec.get("int_upper") if spec.get("int_upper") is not None else [], dtype=int)
    cat_cardinality = np.asarray(spec.get("cat_cardinality") if spec.get("cat_cardinality") is not None else [], dtype=int)
    perm_mutation = _resolve_perm_mutation(spec) if perm_idx.size else None

    if perm_idx.size:
        perm_view = X[:, perm_idx].astype(np.int32, copy=True)
        assert perm_mutation is not None
        perm_mutation(perm_view, prob, rng)
        X[:, perm_idx] = perm_view

    if real_idx.size:
        span = np.maximum(real_upper - real_lower, 1e-6)
        noise = rng.normal(scale=0.1 * span, size=(X.shape[0], real_idx.size))
        mask = rng.random((X.shape[0], real_idx.size)) <= prob
        X_real = X[:, real_idx]
        proposed = X_real + noise
        proposed = np.clip(proposed, real_lower, real_upper)
        X_real = np.where(mask, proposed, X_real)
        X[:, real_idx] = X_real

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


def mixed_initialize(
    pop_size: int,
    n_var: int,
    spec: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Initialize a mixed-typed population using provided index spec.
    """
    if pop_size <= 0 or n_var <= 0:
        raise ValueError("pop_size and n_var must be positive.")
    _validate_mixed_spec(spec, n_var)
    perm_idx = _extract_index_array(spec, "perm_idx")
    real_idx = _extract_index_array(spec, "real_idx")
    int_idx = _extract_index_array(spec, "int_idx")
    cat_idx = _extract_index_array(spec, "cat_idx")
    real_lower = np.asarray(spec.get("real_lower") if spec.get("real_lower") is not None else [], dtype=float)
    real_upper = np.asarray(spec.get("real_upper") if spec.get("real_upper") is not None else [], dtype=float)
    int_lower = np.asarray(spec.get("int_lower") if spec.get("int_lower") is not None else [], dtype=int)
    int_upper = np.asarray(spec.get("int_upper") if spec.get("int_upper") is not None else [], dtype=int)
    cat_cardinality = np.asarray(spec.get("cat_cardinality") if spec.get("cat_cardinality") is not None else [], dtype=int)

    X = np.zeros((pop_size, n_var), dtype=float)
    if perm_idx.size:
        keys = rng.random((pop_size, perm_idx.size))
        perms = np.argsort(keys, axis=1).astype(np.int32, copy=False)
        X[:, perm_idx] = perms
    if real_idx.size:
        X[:, real_idx] = rng.uniform(real_lower, real_upper, size=(pop_size, real_idx.size))
    if int_idx.size:
        X[:, int_idx] = rng.integers(int_lower, int_upper + 1, size=(pop_size, int_idx.size), dtype=np.int32)
    if cat_idx.size:
        cats = [rng.integers(0, int(cat_cardinality[i]), size=pop_size, dtype=np.int32) for i in range(cat_idx.size)]
        X[:, cat_idx] = np.stack(cats, axis=1)
    return X


def mixed_crossover(
    X_parents: np.ndarray,
    prob: float,
    spec: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Mixed crossover: permutation crossover for perm_idx, arithmetic mean for real,
    uniform swap for int/cat.

    Optional segment-level overrides in ``spec``:
      - ``*_crossover_prob`` per segment (perm/real/int/cat)
      - ``int_crossover`` in {"uniform", "arithmetic", "sbx"}
      - ``int_crossover_eta`` for integer SBX
      - ``real_crossover`` in {"arithmetic", "mean"}
      - ``cat_crossover`` in {"uniform"}
    """
    if not _has_customized_segment_settings(spec, _CUSTOM_CROSSOVER_KEYS):
        return _mixed_crossover_legacy(X_parents, prob, spec, rng)

    Np, D = X_parents.shape
    if Np == 0:
        return np.empty_like(X_parents)
    n_original = Np
    if Np % 2 != 0:
        X_parents = np.vstack([X_parents, X_parents[-1:]])
        Np += 1
    pairs = X_parents.reshape(Np // 2, 2, D).copy()
    _validate_mixed_spec(spec, D)

    perm_idx = _extract_index_array(spec, "perm_idx")
    real_idx = _extract_index_array(spec, "real_idx")
    int_idx = _extract_index_array(spec, "int_idx")
    cat_idx = _extract_index_array(spec, "cat_idx")
    int_lower = np.asarray(spec.get("int_lower") if spec.get("int_lower") is not None else [], dtype=int)
    int_upper = np.asarray(spec.get("int_upper") if spec.get("int_upper") is not None else [], dtype=int)
    _validate_segment_bounds(
        index_name="int_idx",
        idx=int_idx,
        lower_name="int_lower",
        lower=int_lower,
        upper_name="int_upper",
        upper=int_upper,
    )

    perm_prob = _resolve_probability(spec, "perm_crossover_prob", prob)
    real_prob = _resolve_probability(spec, "real_crossover_prob", prob)
    int_prob = _resolve_probability(spec, "int_crossover_prob", prob)
    cat_prob = _resolve_probability(spec, "cat_crossover_prob", prob)

    real_method = _resolve_choice(spec, "real_crossover", "arithmetic", allowed={"arithmetic", "mean"})
    int_method = _resolve_choice(spec, "int_crossover", "uniform", allowed={"uniform", "arithmetic", "sbx", "integer_sbx"})
    cat_method = _resolve_choice(spec, "cat_crossover", "uniform", allowed={"uniform"})
    int_eta = float(spec.get("int_crossover_eta", 20.0))

    perm_crossover = _resolve_perm_crossover(spec) if perm_idx.size else None

    if perm_idx.size and perm_prob > 0.0:
        active = np.flatnonzero(rng.random(pairs.shape[0]) <= perm_prob)
        for row in active:
            parents_perm = np.stack([pairs[row, 0, perm_idx], pairs[row, 1, perm_idx]], axis=0).astype(np.int32, copy=True)
            assert perm_crossover is not None
            perm_children = perm_crossover(parents_perm, 1.0, rng)
            pairs[row, 0, perm_idx] = perm_children[0]
            pairs[row, 1, perm_idx] = perm_children[1]

    if real_idx.size and real_prob > 0.0:
        active = np.flatnonzero(rng.random(pairs.shape[0]) <= real_prob)
        if real_method in {"arithmetic", "mean"}:
            for row in active:
                mean_vals = 0.5 * (pairs[row, 0, real_idx] + pairs[row, 1, real_idx])
                pairs[row, 0, real_idx] = mean_vals
                pairs[row, 1, real_idx] = mean_vals
        else:
            raise ValueError(f"Unsupported real_crossover '{real_method}'.")

    if int_idx.size and int_prob > 0.0:
        active = np.flatnonzero(rng.random(pairs.shape[0]) <= int_prob)
        for row in active:
            p1 = pairs[row, 0, int_idx]
            p2 = pairs[row, 1, int_idx]
            if int_method == "uniform":
                swap_mask = rng.random(int_idx.size) < 0.5
                if np.any(swap_mask):
                    tmp = p1[swap_mask].copy()
                    p1[swap_mask] = p2[swap_mask]
                    p2[swap_mask] = tmp
            elif int_method == "arithmetic":
                mean_vals = np.rint(0.5 * (p1 + p2))
                p1[:] = mean_vals
                p2[:] = mean_vals
            elif int_method in {"sbx", "integer_sbx"}:
                pair_block = np.stack([p1, p2], axis=0).astype(float, copy=True)
                children = integer_sbx_crossover(
                    pair_block,
                    prob=1.0,
                    eta=int_eta,
                    lower=int_lower,
                    upper=int_upper,
                    rng=rng,
                )
                p1[:] = children[0]
                p2[:] = children[1]
            else:
                raise ValueError(f"Unsupported int_crossover '{int_method}'.")
            pairs[row, 0, int_idx] = p1
            pairs[row, 1, int_idx] = p2

    if cat_idx.size and cat_prob > 0.0:
        if cat_method != "uniform":
            raise ValueError(f"Unsupported cat_crossover '{cat_method}'.")
        active = np.flatnonzero(rng.random(pairs.shape[0]) <= cat_prob)
        for row in active:
            swap_mask = rng.random(cat_idx.size) < 0.5
            if not np.any(swap_mask):
                continue
            cols = cat_idx[swap_mask]
            tmp = pairs[row, 0, cols].copy()
            pairs[row, 0, cols] = pairs[row, 1, cols]
            pairs[row, 1, cols] = tmp

    offspring = pairs.reshape(Np, D)
    if n_original % 2 != 0:
        offspring = offspring[:n_original]
    return offspring


def mixed_mutation(
    X: np.ndarray,
    prob: float,
    spec: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    """
    Mixed mutation: permutation mutation for perm_idx, Gaussian perturb for real,
    random reset for int/cat.

    Optional segment-level overrides in ``spec``:
      - ``*_mutation_prob`` per segment (perm/real/int/cat)
      - ``int_mutation`` in {"reset", "creep", "pm", "polynomial"}
      - ``int_mutation_step`` for creep, ``int_mutation_eta`` for polynomial
      - ``real_mutation`` in {"gaussian", "reset", "pm", "polynomial"}
      - ``real_mutation_sigma`` or ``real_mutation_sigma_factor``
      - ``cat_mutation`` in {"reset"}
    """
    if not _has_customized_segment_settings(spec, _CUSTOM_MUTATION_KEYS):
        _mixed_mutation_legacy(X, prob, spec, rng)
        return

    if X.size == 0:
        return

    _validate_mixed_spec(spec, X.shape[1])
    perm_idx = _extract_index_array(spec, "perm_idx")
    real_idx = _extract_index_array(spec, "real_idx")
    int_idx = _extract_index_array(spec, "int_idx")
    cat_idx = _extract_index_array(spec, "cat_idx")
    real_lower = np.asarray(spec.get("real_lower") if spec.get("real_lower") is not None else [], dtype=float)
    real_upper = np.asarray(spec.get("real_upper") if spec.get("real_upper") is not None else [], dtype=float)
    int_lower = np.asarray(spec.get("int_lower") if spec.get("int_lower") is not None else [], dtype=int)
    int_upper = np.asarray(spec.get("int_upper") if spec.get("int_upper") is not None else [], dtype=int)
    cat_cardinality = np.asarray(spec.get("cat_cardinality") if spec.get("cat_cardinality") is not None else [], dtype=int)

    _validate_segment_bounds(
        index_name="real_idx",
        idx=real_idx,
        lower_name="real_lower",
        lower=real_lower,
        upper_name="real_upper",
        upper=real_upper,
    )
    _validate_segment_bounds(
        index_name="int_idx",
        idx=int_idx,
        lower_name="int_lower",
        lower=int_lower,
        upper_name="int_upper",
        upper=int_upper,
    )
    if cat_idx.size and cat_cardinality.shape[0] != cat_idx.size:
        raise ValueError("cat_cardinality length must match cat_idx size.")

    perm_prob = _resolve_probability(spec, "perm_mutation_prob", prob)
    real_prob = _resolve_probability(spec, "real_mutation_prob", prob)
    int_prob = _resolve_probability(spec, "int_mutation_prob", prob)
    cat_prob = _resolve_probability(spec, "cat_mutation_prob", prob)

    real_method = _resolve_choice(
        spec,
        "real_mutation",
        "gaussian",
        allowed={"gaussian", "reset", "uniform_reset", "random_reset", "pm", "polynomial"},
    )
    int_method = _resolve_choice(
        spec,
        "int_mutation",
        "reset",
        allowed={"reset", "random_reset", "creep", "pm", "polynomial"},
    )
    cat_method = _resolve_choice(
        spec,
        "cat_mutation",
        "reset",
        allowed={"reset", "uniform_reset", "random_reset"},
    )

    if perm_idx.size and perm_prob > 0.0:
        perm_mutation = _resolve_perm_mutation(spec)
        perm_view = X[:, perm_idx].astype(np.int32, copy=True)
        perm_mutation(perm_view, perm_prob, rng)
        X[:, perm_idx] = perm_view

    if real_idx.size and real_prob > 0.0:
        X_real = np.asarray(X[:, real_idx], dtype=float)
        if real_method == "gaussian":
            sigma_raw = spec.get("real_mutation_sigma")
            if sigma_raw is None:
                sigma_factor = float(spec.get("real_mutation_sigma_factor", 0.1))
                span = np.maximum(real_upper - real_lower, 1.0e-6)
                sigma: float | np.ndarray = sigma_factor * span
            else:
                sigma_arr = np.asarray(sigma_raw, dtype=float)
                if sigma_arr.ndim == 0:
                    sigma = float(sigma_arr)
                else:
                    if sigma_arr.shape[0] != real_idx.size:
                        raise ValueError("real_mutation_sigma length must match real_idx size.")
                    sigma = sigma_arr
            noise = rng.normal(scale=sigma, size=(X.shape[0], real_idx.size))
            mask = rng.random((X.shape[0], real_idx.size)) <= real_prob
            proposed = np.clip(X_real + noise, real_lower, real_upper)
            X_real = np.where(mask, proposed, X_real)
        elif real_method in {"reset", "uniform_reset", "random_reset"}:
            mask = rng.random((X.shape[0], real_idx.size)) <= real_prob
            resampled = rng.uniform(real_lower, real_upper, size=(X.shape[0], real_idx.size))
            X_real[mask] = resampled[mask]
        elif real_method in {"pm", "polynomial"}:
            eta = float(spec.get("real_mutation_eta", 20.0))
            if eta <= 0.0:
                raise ValueError("real_mutation_eta must be > 0.")
            rnd_mask = rng.random((X.shape[0], real_idx.size))
            rnd_delta = rng.random((X.shape[0], real_idx.size))
            mask = rnd_mask <= real_prob
            if np.any(mask):
                mut_pow = 1.0 / (eta + 1.0)
                rows, cols = np.nonzero(mask)
                for i, j in zip(rows, cols):
                    y = float(X_real[i, j])
                    yl = float(real_lower[j])
                    yu = float(real_upper[j])
                    if yu <= yl:
                        continue
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    rnd = float(rnd_delta[i, j])
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (eta + 1.0))
                        deltaq = val**mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (eta + 1.0))
                        deltaq = 1.0 - val**mut_pow
                    y = y + deltaq * (yu - yl)
                    X_real[i, j] = min(max(y, yl), yu)
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
        elif int_method in {"pm", "polynomial"}:
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
            for j in range(cat_idx.size):
                cats[:, j] = rng.integers(0, int(cat_cardinality[j]), size=X.shape[0], dtype=np.int32)
            X_cat = np.rint(X[:, cat_idx]).astype(np.int32, copy=True)
            X_cat[mask] = cats[mask]
            X[:, cat_idx] = X_cat


# === Adapters ===


class MixedCrossover:
    def __init__(self, prob: float = 0.9, **kwargs: Any) -> None:
        self.prob = float(prob)

    def __call__(self, parents: np.ndarray, rng: np.random.Generator, **kwargs: Any) -> np.ndarray:
        spec = kwargs.get("spec")
        if spec is None:
            raise ValueError("MixedCrossover requires 'spec' in kwargs.")
        return mixed_crossover(parents, self.prob, spec, rng)


class MixedMutation:
    def __init__(self, prob: float = 0.1, **kwargs: Any) -> None:
        self.prob = float(prob)

    def __call__(self, X: np.ndarray, rng: np.random.Generator, **kwargs: Any) -> None:
        spec = kwargs.get("spec")
        if spec is None:
            raise ValueError("MixedMutation requires 'spec' in kwargs.")
        mixed_mutation(X, self.prob, spec, rng)


__all__ = [
    "mixed_initialize",
    "mixed_crossover",
    "mixed_mutation",
    "MixedCrossover",
    "MixedMutation",
]
