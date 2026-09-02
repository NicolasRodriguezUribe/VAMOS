"""Bitwise logical-array comparison for exact replay."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .reports import ArrayComparison

MANDATORY_EXACT_ARRAYS = ("F", "X")


def compare_array_collections(
    stored: Mapping[str, np.ndarray[Any, Any]],
    replay: Mapping[str, np.ndarray[Any, Any]],
) -> tuple[ArrayComparison, ...]:
    """Compare every role in either collection with mandatory F/X evidence."""
    roles = sorted(set(stored) | set(replay) | set(MANDATORY_EXACT_ARRAYS))
    return tuple(_compare_array(role, stored.get(role), replay.get(role)) for role in roles)


def comparisons_are_exact(comparisons: tuple[ArrayComparison, ...]) -> bool:
    by_role = {item.role: item for item in comparisons}
    return all(item.exact for item in comparisons) and all(by_role.get(role) is not None for role in MANDATORY_EXACT_ARRAYS)


def _compare_array(role: str, stored: np.ndarray[Any, Any] | None, replay: np.ndarray[Any, Any] | None) -> ArrayComparison:
    stored_bytes = _logical_bytes(stored)
    replay_bytes = _logical_bytes(replay)
    mismatch = _mismatch(stored, replay, stored_bytes, replay_bytes)
    return ArrayComparison(
        role=role,
        exact=mismatch is None,
        stored_dtype=stored.dtype.str if stored is not None else None,
        replay_dtype=replay.dtype.str if replay is not None else None,
        stored_shape=tuple(int(value) for value in stored.shape) if stored is not None else None,
        replay_shape=tuple(int(value) for value in replay.shape) if replay is not None else None,
        stored_sha256=_hash(stored_bytes),
        replay_sha256=_hash(replay_bytes),
        first_difference=_first_difference(stored, replay, stored_bytes, replay_bytes),
        maximum_absolute_difference=_maximum_absolute_difference(stored, replay),
        mismatch=mismatch,
    )


def _mismatch(
    stored: np.ndarray[Any, Any] | None, replay: np.ndarray[Any, Any] | None, stored_bytes: bytes | None, replay_bytes: bytes | None
) -> str | None:
    if stored is None:
        return "missing_stored_array"
    if replay is None:
        return "missing_replay_array"
    if stored.dtype.str != replay.dtype.str:
        return "dtype_mismatch"
    if stored.shape != replay.shape:
        return "shape_mismatch"
    if stored_bytes != replay_bytes:
        return "content_mismatch"
    return None


def _logical_bytes(value: np.ndarray[Any, Any] | None) -> bytes | None:
    if value is None:
        return None
    return np.ascontiguousarray(value).tobytes(order="C")


def _hash(value: bytes | None) -> str | None:
    return hashlib.sha256(value).hexdigest() if value is not None else None


def _first_difference(
    stored: np.ndarray[Any, Any] | None,
    replay: np.ndarray[Any, Any] | None,
    stored_bytes: bytes | None,
    replay_bytes: bytes | None,
) -> tuple[int, ...] | None:
    if stored is None or replay is None or stored.dtype.str != replay.dtype.str or stored.shape != replay.shape:
        return None
    if stored_bytes == replay_bytes or stored_bytes is None or replay_bytes is None:
        return None
    left = np.frombuffer(stored_bytes, dtype=np.uint8)
    right = np.frombuffer(replay_bytes, dtype=np.uint8)
    differing = np.flatnonzero(left != right)
    if differing.size == 0 or stored.size == 0:
        return None
    flat_index = int(differing[0]) // stored.dtype.itemsize
    return tuple(int(value) for value in np.unravel_index(flat_index, stored.shape))


def _maximum_absolute_difference(stored: np.ndarray[Any, Any] | None, replay: np.ndarray[Any, Any] | None) -> float | None:
    if stored is None or replay is None or stored.shape != replay.shape or stored.size == 0:
        return None
    if stored.dtype.kind not in "biuf" or replay.dtype.kind not in "biuf":
        return None
    with np.errstate(all="ignore"):
        difference = np.abs(stored.astype(np.longdouble) - replay.astype(np.longdouble))
    finite = difference[np.isfinite(difference)]
    if finite.size == 0:
        return None
    maximum = float(np.max(finite))
    return maximum if math.isfinite(maximum) else None


__all__ = ["MANDATORY_EXACT_ARRAYS", "compare_array_collections", "comparisons_are_exact"]
