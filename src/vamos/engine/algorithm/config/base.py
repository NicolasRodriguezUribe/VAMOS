"""Base utilities for algorithm configuration."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, Tuple


class _SerializableConfig:
    """Mixin to serialize dataclass configs."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def _require_fields(cfg: Dict[str, Any], fields: Tuple[str, ...], name: str) -> None:
    """Validate that required fields are present in configuration."""
    missing = [field for field in fields if field not in cfg]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{name} configuration missing required fields: {joined}")
