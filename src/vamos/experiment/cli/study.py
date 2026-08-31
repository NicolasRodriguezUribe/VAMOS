"""CLI for read-only durable-study planning."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from vamos.experiment.artifacts.errors import ArtifactResourceLimitError, DuplicateJSONKeyError, RunArtifactError
from vamos.experiment.artifacts.jsonio import load_json_file
from vamos.experiment.study.errors import InvalidStudySpecError, StudyError
from vamos.experiment.study.models import OnErrorPolicy, StudySpec
from vamos.experiment.study.preflight import (
    PLAN_RESULT_DOCUMENT_TYPE,
    PLAN_RESULT_SCHEMA_VERSION,
    StudyPlanReport,
    plan_study,
)

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


def run_study(argv: Sequence[str] | None = None) -> None:
    """Parse and run the ``vamos study`` command group."""
    parser = argparse.ArgumentParser(prog="vamos study", description="Read-only planning for durable studies.")
    commands = parser.add_subparsers(dest="study_command", required=True)
    plan_parser = commands.add_parser("plan", help="Resolve and explain a StudySpec JSON file without writing or executing.")
    plan_parser.add_argument("config", help="Path to a StudySpec JSON object.")
    plan_parser.add_argument("--output", help="Proposed study directory to inspect without reserving or creating it.")
    plan_parser.add_argument("--json", action="store_true", dest="json_output", help="Emit one stable JSON result document.")
    args = parser.parse_args(argv)
    if args.study_command != "plan":
        raise AssertionError("argparse accepted an unknown study command")
    _run_plan(config=Path(args.config), output=args.output, json_output=bool(args.json_output))


def _run_plan(*, config: Path, output: str | None, json_output: bool) -> None:
    try:
        spec = _load_spec(config)
        report = plan_study(spec, output=output)
    except StudyError as exc:
        if json_output:
            print(json.dumps(_error_envelope(exc, output=output), sort_keys=True, separators=(",", ":")))
        else:
            print(f"Error [{exc.category}]: {exc}", file=sys.stderr)
        raise SystemExit(_error_exit_code(exc)) from None
    if json_output:
        print(json.dumps(report.as_dict(), sort_keys=True, separators=(",", ":")))
    else:
        print(_render_report(report))
    if report.status == "blocked":
        raise SystemExit(5)


def _load_spec(path: Path) -> StudySpec:
    try:
        value = load_json_file(
            path,
            operation="plan study",
            artifact_role="study_plan_input",
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
            operation="plan study",
            reason="UNKNOWN_FIELD" if unknown else "MISSING_FIELD",
            field="$",
            path=path,
            expected=f"required {_REQUIRED_SPEC_FIELDS!r}; optional {_SPEC_FIELDS - _REQUIRED_SPEC_FIELDS!r}",
            actual={"missing": missing, "unknown": unknown},
            action="Use the single documented StudySpec JSON field set; no study was created.",
        )
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
        operation="plan study",
        reason=reason,
        field=field,
        path=path,
        expected=expected,
        actual=actual,
        action="Correct the StudySpec JSON input; no study was created and no task was executed.",
    )


def _error_envelope(exc: StudyError, *, output: str | None) -> dict[str, object]:
    error = exc.as_dict()
    error["filesystem_write_occurred"] = False
    return {
        "document_type": PLAN_RESULT_DOCUMENT_TYPE,
        "schema_version": PLAN_RESULT_SCHEMA_VERSION,
        "operation": "study plan",
        "status": "invalid",
        "valid": False,
        "execution_occurred": False,
        "filesystem_write_occurred": False,
        "plan_id": None,
        "task_ids": [],
        "task_count": 0,
        "total_evaluation_budget": 0,
        "components": {"problems": [], "algorithms": [], "operators": [], "backends": []},
        "seeds": [],
        "population_sizes": [],
        "termination_categories": [],
        "failure_policy": None,
        "reconstructable": False,
        "duplicate_tasks": exc.reason == "DUPLICATE_CANONICAL_TASK",
        "output": {
            "requested_path": output,
            "status": "not_checked",
            "available": None,
            "collision": False,
            "advisory": None,
        },
        "warnings": [],
        "errors": [error],
        "next_actions": [exc.action],
    }


def _render_report(report: StudyPlanReport) -> str:
    output = report.output
    lines = [
        f"Study plan: {report.status}",
        f"Plan ID: {report.plan_id}",
        f"Tasks / total evaluation budget: {report.task_count} / {report.total_evaluation_budget}",
        f"Problems: {', '.join(report.problem_ids) or '-'}",
        f"Algorithms: {', '.join(report.algorithm_ids) or '-'}",
        f"Backends: {', '.join(report.backend_ids) or '-'}",
        f"Seeds: {', '.join(str(seed) for seed in report.seeds) or '-'}",
        f"Population sizes: {', '.join(str(size) for size in report.population_sizes) or '-'}",
        f"Failure policy: {report.failure_policy}",
        f"Output: {output.status} ({output.requested_path or 'not supplied'})",
        "Optimization executed: no",
        "Filesystem writes: no",
    ]
    if report.errors:
        lines.append("Blocking issues:")
        lines.extend(f"  {item.reason}: {item.action}" for item in report.errors)
    if report.warnings:
        lines.append("Warnings:")
        lines.extend(f"  {warning}" for warning in report.warnings)
    lines.extend(f"Next: {action}" for action in report.next_actions)
    return "\n".join(lines)


def _error_exit_code(exc: StudyError) -> int:
    if isinstance(exc, InvalidStudySpecError):
        return 2
    return 2


__all__ = ["run_study"]
