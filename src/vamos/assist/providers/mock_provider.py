from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from vamos.experiment.cli.quickstart import generate_quickstart_config

from .protocol import PlanResponse, ProviderResponse, QuestionsResponse


def _select_template(templates: list[str]) -> str:
    if not templates:
        raise RuntimeError("No quickstart templates are available.")
    if "demo" in templates:
        return "demo"
    return templates[0]


def _select_problem_type(problem_type_hint: str | None) -> Literal["real", "int", "binary"]:
    if problem_type_hint is None:
        return "real"
    normalized = problem_type_hint.strip().lower()
    if normalized == "real":
        return "real"
    if normalized == "int":
        return "int"
    if normalized == "binary":
        return "binary"
    return "real"


def _allowed_default_keys(template: str) -> set[str]:
    config = generate_quickstart_config(template=template, overrides=None)
    defaults = config.get("defaults")
    if not isinstance(defaults, Mapping):
        return set()
    return set(defaults.keys())


class MockPlanProvider:
    name = "mock"

    def propose(
        self,
        prompt: str,
        catalog: Mapping[str, object],
        templates: list[str],
        problem_type_hint: str | None = None,
        answers: dict[str, str] | None = None,
    ) -> ProviderResponse:
        del prompt
        del catalog
        del answers
        template = _select_template(templates)
        overrides: dict[str, object] = {}
        if "max_evaluations" in _allowed_default_keys(template):
            overrides["max_evaluations"] = 123
        return PlanResponse(
            kind="plan",
            template=template,
            problem_type=_select_problem_type(problem_type_hint),
            overrides=overrides,
            warnings=["mock provider used"],
        )


class QuestionsMockProvider:
    name = "mock_questions"

    def propose(
        self,
        prompt: str,
        catalog: Mapping[str, object],
        templates: list[str],
        problem_type_hint: str | None = None,
        answers: dict[str, str] | None = None,
    ) -> ProviderResponse:
        del prompt
        del catalog
        if answers is None:
            return QuestionsResponse(
                kind="questions",
                questions=[
                    "What problem family should this cover?",
                    "What budget should be prioritized?",
                ],
                warnings=["mock provider requested clarifications"],
            )
        template = _select_template(templates)
        overrides: dict[str, object] = {}
        if "max_evaluations" in _allowed_default_keys(template):
            overrides["max_evaluations"] = 123
        return PlanResponse(
            kind="plan",
            template=template,
            problem_type=_select_problem_type(problem_type_hint),
            overrides=overrides,
            warnings=["mock provider used after questions"],
        )


__all__ = ["MockPlanProvider", "QuestionsMockProvider"]
