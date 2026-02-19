from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class AblationVariant:
    name: str
    label: str | None = None
    tags: tuple[str, ...] = ()
    config_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "config_overrides", dict(self.config_overrides))

    def apply(self, base_config: Mapping[str, Any] | None) -> dict[str, Any]:
        return _deep_merge(base_config or {}, self.config_overrides)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label or self.name,
            "tags": list(self.tags),
            "config_overrides": dict(self.config_overrides),
        }


@dataclass(frozen=True)
class AblationTask:
    problem: str
    variant: AblationVariant
    seed: int
    max_evals: int
    engine: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def task_id(self) -> str:
        return f"{self.problem}/{self.variant.name}/seed_{self.seed}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "problem": self.problem,
            "variant": self.variant.as_dict(),
            "seed": self.seed,
            "max_evals": self.max_evals,
            "engine": self.engine,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AblationPlan:
    tasks: tuple[AblationTask, ...]
    problems: tuple[str, ...]
    variants: tuple[AblationVariant, ...]
    seeds: tuple[int, ...]
    default_max_evals: int
    engine: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "problems", tuple(self.problems))
        object.__setattr__(self, "variants", tuple(self.variants))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "default_max_evals": self.default_max_evals,
            "engine": self.engine,
            "problems": list(self.problems),
            "variants": [variant.as_dict() for variant in self.variants],
            "seeds": list(self.seeds),
            "tasks": [task.as_dict() for task in self.tasks],
            "metadata": dict(self.metadata),
        }
