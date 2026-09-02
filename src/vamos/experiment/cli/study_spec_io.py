"""Bounded JSON input translation for durable-study CLI commands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from vamos.experiment.artifacts.errors import ArtifactResourceLimitError, DuplicateJSONKeyError, RunArtifactError
from vamos.experiment.artifacts.jsonio import load_json_file
from vamos.experiment.study.errors import InvalidStudySpecError
from vamos.experiment.study.models import OnErrorPolicy, StudySpec

_SPEC_FIELDS = {
    "problems",
    "algorithms",
    "seeds",
    "max_evaluations",
    "pop_size",
    "engine",
    "eval_strategy",
    "n_var",
    "n_obj",
    "problem_kwargs",
    "algorithm_configs",
    "on_error",
    "max_attempts_per_task",
    "labels",
    "metadata",
}
_REQUIRED_SPEC_FIELDS = {"problems", "algorithms", "seeds"}


def load_study_spec(path: Path) -> StudySpec:
    """Load one closed StudySpec JSON object without executing components."""
    try:
        value = load_json_file(
            path,
            operation="read study config",
            artifact_role="study_config_input",
            max_bytes=8 * 1024 * 1024,
            max_depth=64,
        )
    except DuplicateJSONKeyError as exc:
        raise _input_error(path, "DUPLICATE_JSON_KEY", exc.field, exc.expected, exc.actual) from exc
    except ArtifactResourceLimitError as exc:
        raise _input_error(path, "RESOURCE_LIMIT", exc.field, exc.expected, exc.actual) from exc
    except RunArtifactError as exc:
        reason = "NON_FINITE_NUMBER" if "non-finite" in exc.reason else "MALFORMED_JSON"
        raise _input_error(path, reason, exc.field, exc.expected, exc.actual) from exc
    fields = set(value)
    missing = sorted(_REQUIRED_SPEC_FIELDS - fields)
    unknown = sorted(fields - _SPEC_FIELDS)
    if missing or unknown:
        raise InvalidStudySpecError(
            operation="read study config",
            reason="UNKNOWN_FIELD" if unknown else "MISSING_FIELD",
            field="$",
            path=path,
            expected=f"required {_REQUIRED_SPEC_FIELDS!r}; optional {_SPEC_FIELDS - _REQUIRED_SPEC_FIELDS!r}",
            actual={"missing": missing, "unknown": unknown},
            action="Use the documented StudySpec JSON fields; no study was changed.",
        )
    return _construct_spec(value)


def _construct_spec(value: Mapping[str, object]) -> StudySpec:
    return StudySpec(
        problems=cast(Sequence[str], value["problems"]),
        algorithms=cast(Sequence[str], value["algorithms"]),
        seeds=cast(Sequence[int], value["seeds"]),
        max_evaluations=cast(int | None, value.get("max_evaluations")),
        pop_size=cast(int | None, value.get("pop_size")),
        engine=cast(str | None, value.get("engine")),
        eval_strategy=cast(str, value.get("eval_strategy", "serial")),
        n_var=cast(int | None, value.get("n_var")),
        n_obj=cast(int | None, value.get("n_obj")),
        problem_kwargs=cast(Mapping[str, object] | None, value.get("problem_kwargs")),
        algorithm_configs=cast(Mapping[str, object] | None, value.get("algorithm_configs")),
        on_error=cast(OnErrorPolicy, value.get("on_error", "fail_fast")),
        max_attempts_per_task=cast(int, value.get("max_attempts_per_task", 3)),
        labels=cast(Mapping[str, object] | None, value.get("labels")),
        metadata=cast(Mapping[str, object] | None, value.get("metadata")),
    )


def _input_error(path: Path, reason: str, field: str | None, expected: object, actual: object) -> InvalidStudySpecError:
    return InvalidStudySpecError(
        operation="read study config",
        reason=reason,
        field=field,
        path=path,
        expected=expected,
        actual=actual,
        action="Correct the StudySpec JSON input; no study was changed and no task was executed.",
    )


__all__ = ["load_study_spec"]
