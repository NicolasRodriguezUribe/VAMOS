from __future__ import annotations

from typing import Any


def spec_default(experiment_defaults: dict[str, Any], key: str, fallback: Any) -> Any:
    """Return the spec-provided default if available, else the fallback."""
    return experiment_defaults.get(key, fallback)
