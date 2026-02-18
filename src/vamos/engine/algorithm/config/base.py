"""Base utilities for algorithm configuration."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict
from typing import Any, Literal, Self, cast

# ── Shared Literal type aliases ──────────────────────────────────────
ResultMode = Literal["non_dominated", "population"]
ConstraintModeStr = Literal["none", "feasibility", "penalty", "epsilon"]
LiveCallbackMode = Literal["nd_only", "all"]
IndicatorType = Literal["eps", "hypervolume"]


class _SerializableConfig:
    """Mixin to serialize dataclass configs."""

    __dataclass_fields__: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(cast(Any, self)))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Self:
        """Reconstruct a config from a serialised dict (inverse of ``to_dict()``)."""
        field_names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        filtered: dict[str, object] = {}
        for key, value in data.items():
            if key not in field_names:
                continue
            # Reconstruct nested ExternalArchiveConfig from plain dict
            if key == "external_archive" and isinstance(value, dict):
                from vamos.archive import ExternalArchiveConfig
                value = ExternalArchiveConfig(**value)
            filtered[key] = value
        return cls(**filtered)  # type: ignore[call-arg]


def _require_fields(cfg: dict[str, Any], fields: tuple[str, ...], name: str) -> None:
    """Validate that required fields are present in configuration."""
    missing = [field for field in fields if field not in cfg]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{name} configuration missing required fields: {joined}")
