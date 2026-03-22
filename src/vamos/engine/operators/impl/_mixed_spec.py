from __future__ import annotations

from typing import Any

import numpy as np

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

PERM_CROSSOVER = {
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

PERM_MUTATION = {
    "swap": swap_mutation,
    "insert": insert_mutation,
    "scramble": scramble_mutation,
    "inversion": inversion_mutation,
    "displacement": displacement_mutation,
    "two_opt": two_opt_mutation,
}

CUSTOM_CROSSOVER_KEYS: set[str] = {
    "perm_crossover_prob",
    "real_crossover_prob",
    "int_crossover_prob",
    "cat_crossover_prob",
    "real_crossover",
    "int_crossover",
    "cat_crossover",
    "int_crossover_eta",
}

CUSTOM_MUTATION_KEYS: set[str] = {
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


def extract_index_array(spec: dict[str, Any], key: str) -> np.ndarray:
    raw = spec.get(key)
    if raw is None:
        return np.asarray([], dtype=int)
    return np.asarray(raw, dtype=int)


def validate_mixed_spec(spec: dict[str, Any], n_var: int) -> None:
    indices = {
        "perm_idx": extract_index_array(spec, "perm_idx"),
        "real_idx": extract_index_array(spec, "real_idx"),
        "int_idx": extract_index_array(spec, "int_idx"),
        "cat_idx": extract_index_array(spec, "cat_idx"),
    }
    if not any(idx.size for idx in indices.values()):
        return
    all_idx = np.concatenate([idx for idx in indices.values() if idx.size])
    if np.any(all_idx < 0) or np.any(all_idx >= n_var):
        raise ValueError("mixed_spec indices must be within [0, n_var).")
    if np.unique(all_idx).size != all_idx.size:
        raise ValueError("mixed_spec indices must be disjoint across segments.")


def resolve_perm_crossover(spec: dict[str, Any]) -> Any:
    method = str(spec.get("perm_crossover", "ox")).lower()
    try:
        return PERM_CROSSOVER[method]
    except KeyError as exc:
        available = ", ".join(sorted(PERM_CROSSOVER))
        raise ValueError(f"Unknown perm_crossover '{method}'. Available: {available}") from exc


def resolve_perm_mutation(spec: dict[str, Any]) -> Any:
    method = str(spec.get("perm_mutation", "swap")).lower()
    try:
        return PERM_MUTATION[method]
    except KeyError as exc:
        available = ", ".join(sorted(PERM_MUTATION))
        raise ValueError(f"Unknown perm_mutation '{method}'. Available: {available}") from exc


def resolve_probability(spec: dict[str, Any], key: str, fallback: float) -> float:
    raw = spec.get(key, fallback)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{key}' must be a real number in [0,1].") from exc
    return float(np.clip(value, 0.0, 1.0))


def resolve_choice(spec: dict[str, Any], key: str, default: str, *, allowed: set[str]) -> str:
    raw = str(spec.get(key, default)).strip().lower()
    if raw not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"Unknown {key} '{raw}'. Available: {options}")
    return raw


def has_customized_segment_settings(spec: dict[str, Any], keys: set[str]) -> bool:
    return any(key in spec for key in keys)


def validate_segment_bounds(
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


__all__ = [
    "CUSTOM_CROSSOVER_KEYS",
    "CUSTOM_MUTATION_KEYS",
    "extract_index_array",
    "has_customized_segment_settings",
    "resolve_choice",
    "resolve_perm_crossover",
    "resolve_perm_mutation",
    "resolve_probability",
    "validate_mixed_spec",
    "validate_segment_bounds",
]
