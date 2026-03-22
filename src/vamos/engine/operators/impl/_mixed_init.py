from __future__ import annotations

from typing import Any

import numpy as np

from ._mixed_spec import extract_index_array, validate_mixed_spec


def mixed_initialize(pop_size: int, n_var: int, spec: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Initialize a mixed-typed population using provided index spec."""
    if pop_size <= 0 or n_var <= 0:
        raise ValueError("pop_size and n_var must be positive.")
    validate_mixed_spec(spec, n_var)
    perm_idx = extract_index_array(spec, "perm_idx")
    real_idx = extract_index_array(spec, "real_idx")
    int_idx = extract_index_array(spec, "int_idx")
    cat_idx = extract_index_array(spec, "cat_idx")
    real_lower = np.asarray(spec.get("real_lower") if spec.get("real_lower") is not None else [], dtype=float)
    real_upper = np.asarray(spec.get("real_upper") if spec.get("real_upper") is not None else [], dtype=float)
    int_lower = np.asarray(spec.get("int_lower") if spec.get("int_lower") is not None else [], dtype=int)
    int_upper = np.asarray(spec.get("int_upper") if spec.get("int_upper") is not None else [], dtype=int)
    cat_cardinality = np.asarray(spec.get("cat_cardinality") if spec.get("cat_cardinality") is not None else [], dtype=int)

    X = np.zeros((pop_size, n_var), dtype=float)
    if perm_idx.size:
        X[:, perm_idx] = np.argsort(rng.random((pop_size, perm_idx.size)), axis=1).astype(np.int32, copy=False)
    if real_idx.size:
        X[:, real_idx] = rng.uniform(real_lower, real_upper, size=(pop_size, real_idx.size))
    if int_idx.size:
        X[:, int_idx] = rng.integers(int_lower, int_upper + 1, size=(pop_size, int_idx.size), dtype=np.int32)
    if cat_idx.size:
        cats = [rng.integers(0, int(cat_cardinality[i]), size=pop_size, dtype=np.int32) for i in range(cat_idx.size)]
        X[:, cat_idx] = np.stack(cats, axis=1)
    return X


__all__ = ["mixed_initialize"]
