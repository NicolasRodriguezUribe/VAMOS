from __future__ import annotations

from typing import Literal, TypeAlias

Encoding: TypeAlias = Literal["real", "binary", "permutation", "integer", "mixed"]

EncodingLike: TypeAlias = Encoding | Literal["continuous", "float", "perm", "int"]

ENCODINGS: tuple[Encoding, ...] = ("real", "binary", "permutation", "integer", "mixed")

_ALIASES: dict[str, Encoding] = {
    "continuous": "real",
    "float": "real",
    "real": "real",
    "binary": "binary",
    "permutation": "permutation",
    "perm": "permutation",
    "integer": "integer",
    "int": "integer",
    "mixed": "mixed",
}


def normalize_encoding(value: str | None, *, default: Encoding = "real") -> Encoding:
    """
    Normalize user/problem encoding strings to canonical encoding identifiers.

    Canonical encodings are: "real", "binary", "permutation", "integer", "mixed".
    """
    if value is None:
        return default
    key = value.strip().lower()
    if not key:
        return default
    normalized = _ALIASES.get(key)
    if normalized is None:
        expected = ", ".join(sorted(set(_ALIASES)))
        raise ValueError(f"Unknown encoding '{value}'. Expected one of: {expected}.")
    return normalized


__all__ = ["Encoding", "EncodingLike", "ENCODINGS", "normalize_encoding"]
