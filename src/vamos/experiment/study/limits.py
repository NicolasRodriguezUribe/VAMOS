"""Finite defensive limits for data-only study loading."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class StudyLoadLimits:
    max_manifest_bytes: int = 8 * 1024 * 1024
    max_spec_bytes: int = 8 * 1024 * 1024
    max_plan_bytes: int = 64 * 1024 * 1024
    max_task_bytes: int = 1024 * 1024
    max_attempt_bytes: int = 2 * 1024 * 1024
    max_event_bytes: int = 2 * 1024 * 1024
    max_tasks: int = 100_000
    max_documents: int = 300_000
    max_total_bytes: int = 512 * 1024 * 1024
    max_json_depth: int = 64
    max_string_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"StudyLoadLimits.{item.name} must be a positive integer.")


__all__ = ["StudyLoadLimits"]
