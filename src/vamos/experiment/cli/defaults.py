from __future__ import annotations

from typing import TypeVar, cast

from vamos.engine.config.spec import SpecBlock

T = TypeVar("T")


def spec_default(experiment_defaults: SpecBlock, key: str, fallback: T) -> T:
    """Return the spec-provided default if available, else the fallback."""
    return cast(T, experiment_defaults.get(key, fallback))
