from __future__ import annotations

from typing import Any

import numpy as np

from .permutation import (
    order_crossover,
    pmx_crossover,
    cycle_crossover,
    position_based_crossover,
    edge_recombination_crossover,
    swap_mutation,
    insert_mutation,
    scramble_mutation,
    inversion_mutation,
    displacement_mutation,
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
}

_PERM_MUTATION = {
    "swap": swap_mutation,
    "insert": insert_mutation,
    "scramble": scramble_mutation,
    "inversion": inversion_mutation,
    "displacement": displacement_mutation,
}


def _extract_index_array(spec: dict[str, np.ndarray], key: str) -> np.ndarray:
    raw = spec.get(key)
    if raw is None:
        return np.asarray([], dtype=int)
    return np.asarray(raw, dtype=int)


def _validate_mixed_spec(spec: dict[str, np.ndarray], n_var: int) -> None:
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


def _resolve_perm_crossover(spec: dict[str, np.ndarray]) -> Any:
    method = str(spec.get("perm_crossover", "ox")).lower()
    try:
        return _PERM_CROSSOVER[method]
    except KeyError as exc:
        available = ", ".join(sorted(_PERM_CROSSOVER))
        raise ValueError(f"Unknown perm_crossover '{method}'. Available: {available}") from exc


def _resolve_perm_mutation(spec: dict[str, np.ndarray]) -> Any:
    method = str(spec.get("perm_mutation", "swap")).lower()
    try:
        return _PERM_MUTATION[method]
    except KeyError as exc:
        available = ", ".join(sorted(_PERM_MUTATION))
        raise ValueError(f"Unknown perm_mutation '{method}'. Available: {available}") from exc


def mixed_initialize(
    pop_size: int,
    n_var: int,
    spec: dict[str, np.ndarray],
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
    spec: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Mixed crossover: permutation crossover for perm_idx, arithmetic mean for real,
    uniform swap for int/cat.
    """
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


def mixed_mutation(
    X: np.ndarray,
    prob: float,
    spec: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> None:
    """
    Mixed mutation: permutation mutation for perm_idx, Gaussian perturb for real,
    random reset for int/cat.
    """
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
