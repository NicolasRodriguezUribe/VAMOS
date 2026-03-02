"""Shared repair resolution for real-valued operator policies."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from vamos.engine.algorithm.components.variation.protocol import RepairConfigValue, RepairName, RepairOperator
from vamos.engine.operators.impl.registry import get_operator_registry
from vamos.foundation.encoding import EncodingLike, normalize_encoding

__all__ = ["apply_policy_repair", "resolve_policy_repair"]


def resolve_policy_repair(
    encoding: EncodingLike,
    repair_cfg: RepairConfigValue = "auto",
) -> RepairOperator | None:
    """Resolve a policy-level repair operator using public ``"auto"`` semantics."""
    normalized = normalize_encoding(encoding)
    if normalized != "real":
        if repair_cfg == "auto":
            return None
        raise ValueError("Repair operators are only supported for real encoding.")

    if repair_cfg == "auto":
        method: RepairName = "clip"
        params: dict[str, Any] = {}
    else:
        method, params = repair_cfg

    try:
        op_cls = cast(type[Any], get_operator_registry().get(method.lower()))
    except KeyError as exc:
        available = ", ".join(get_operator_registry().list())
        raise ValueError(f"Unknown repair operator '{method}'. Available: {available}") from exc

    try:
        op = op_cls(**params) if params else op_cls()
    except TypeError as exc:
        raise ValueError(f"Failed to initialize repair '{method}' with params {params}. Error: {exc}") from exc

    return cast(RepairOperator, op)


def apply_policy_repair(
    repair_op: RepairOperator,
    values: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a real repair operator to 2D or grouped 3D offspring arrays."""
    arr = np.asarray(values)
    if arr.ndim == 2:
        return np.asarray(repair_op(arr, xl, xu, rng))
    if arr.ndim == 3:
        repaired = repair_op(arr.reshape(-1, arr.shape[-1]), xl, xu, rng)
        return np.asarray(repaired).reshape(arr.shape)
    raise ValueError("Real repair expects arrays shaped (n_individuals, n_vars) or (n_groups, n_children, n_vars).")
