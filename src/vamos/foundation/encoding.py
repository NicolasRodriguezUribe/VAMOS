from __future__ import annotations

import difflib
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
        all_names = sorted(set(_ALIASES))
        matches = difflib.get_close_matches(key, all_names, n=3, cutoff=0.5)
        if matches:
            raise ValueError(f"Unknown encoding '{value}'. Did you mean: {', '.join(matches)}?")
        expected = ", ".join(all_names)
        raise ValueError(f"Unknown encoding '{value}'. Expected one of: {expected}.")
    return normalized


__all__ = ["Encoding", "EncodingLike", "ENCODINGS", "normalize_encoding"]
