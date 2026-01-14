"""
Static typing helpers for the experiment YAML/JSON spec.

The CLI config format is intentionally dict-based (YAML/JSON friendly) and is
validated at runtime by `spec_validation.py`. These types document the expected
shape for editors and type-checkers without constraining the runtime format.
"""

from __future__ import annotations

from typing import Any, TypedDict


SpecBlock = dict[str, Any]


class ExperimentSpec(TypedDict, total=False):
    version: str
    defaults: SpecBlock
    problems: dict[str, SpecBlock | None]
    stopping: SpecBlock
    archive: SpecBlock


__all__ = ["ExperimentSpec", "SpecBlock"]
