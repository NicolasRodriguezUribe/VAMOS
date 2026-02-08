from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol, TypedDict


class PlanResponse(TypedDict):
    kind: Literal["plan"]
    template: str
    problem_type: Literal["real", "int", "binary"]
    overrides: dict[str, object]
    warnings: list[str]


class QuestionsResponse(TypedDict):
    kind: Literal["questions"]
    questions: list[str]
    warnings: list[str]


ProviderResponse = PlanResponse | QuestionsResponse


class PlanProvider(Protocol):
    name: str

    def propose(
        self,
        prompt: str,
        catalog: Mapping[str, object],
        templates: list[str],
        problem_type_hint: str | None = None,
        answers: dict[str, str] | None = None,
    ) -> ProviderResponse: ...


__all__ = ["PlanProvider", "PlanResponse", "ProviderResponse", "QuestionsResponse"]
