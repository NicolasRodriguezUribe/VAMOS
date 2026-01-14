"""Base utilities for algorithm configuration."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, cast


class _SerializableConfig:
    """Mixin to serialize dataclass configs."""

    __dataclass_fields__: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(cast(Any, self)))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def _require_fields(cfg: dict[str, Any], fields: tuple[str, ...], name: str) -> None:
    """Validate that required fields are present in configuration."""
    missing = [field for field in fields if field not in cfg]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{name} configuration missing required fields: {joined}")
