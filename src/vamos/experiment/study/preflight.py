"""Read-only reporting for deterministic study planning."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from .errors import StudyError
from .loading import load_study
from .models import OnErrorPolicy, ResolvedStudyPlan, StudySpec
from .planning import resolve_spec

PlanStatus = Literal["ready", "blocked"]
OutputStatus = Literal[
    "not_checked",
    "available",
    "existing_file",
    "empty_directory",
    "canonical_study",
    "invalid_study_directory",
    "unrelated_directory",
    "existing_symlink",
    "existing_other",
]


@dataclass(frozen=True, slots=True)
class StudyPlanDiagnostic:
    """One stable, immutable planning diagnostic."""

    operation: str
    reason: str
    field: str | None
    expected: str
    actual: str
    action: str
    execution_occurred: bool = False
    filesystem_write_occurred: bool = False

    def as_dict(self) -> dict[str, object]:
        """Return a detached JSON-ready diagnostic."""
        return {
            "operation": self.operation,
            "reason": self.reason,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "execution_occurred": self.execution_occurred,
            "filesystem_write_occurred": self.filesystem_write_occurred,
            "action": self.action,
        }


@dataclass(frozen=True, slots=True)
class StudyPlanOutput:
    """Advisory, read-only classification of a proposed output path."""

    requested_path: str | None
    status: OutputStatus
    available: bool | None
    collision: bool
    advisory: str | None

    def as_dict(self) -> dict[str, object]:
        """Return a detached JSON-ready output classification."""
        return {
            "requested_path": self.requested_path,
            "status": self.status,
            "available": self.available,
            "collision": self.collision,
            "advisory": self.advisory,
        }


@dataclass(frozen=True, slots=True)
class StudyPlanReport:
    """Immutable explanation of a resolved study with no published state."""

    plan: ResolvedStudyPlan
    status: PlanStatus
    valid: bool
    total_evaluation_budget: int
    problem_ids: tuple[str, ...]
    algorithm_ids: tuple[str, ...]
    operator_ids: tuple[str, ...]
    backend_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    population_sizes: tuple[int, ...]
    termination_categories: tuple[str, ...]
    failure_policy: OnErrorPolicy
    reconstructable: bool
    duplicate_tasks: bool
    output: StudyPlanOutput
    warnings: tuple[str, ...]
    errors: tuple[StudyPlanDiagnostic, ...]
    next_actions: tuple[str, ...]

    @property
    def plan_id(self) -> str:
        return self.plan.plan_id

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.plan.tasks)

    @property
    def task_count(self) -> int:
        return self.plan.task_count

    def as_dict(self) -> dict[str, object]:
        """Return a detached semantic payload for Python and CLI consumers."""
        return {
            "status": self.status,
            "valid": self.valid,
            "execution_occurred": False,
            "filesystem_write_occurred": False,
            "plan_id": self.plan_id,
            "task_ids": list(self.task_ids),
            "task_count": self.task_count,
            "total_evaluation_budget": self.total_evaluation_budget,
            "components": {
                "problems": list(self.problem_ids),
                "algorithms": list(self.algorithm_ids),
                "operators": list(self.operator_ids),
                "backends": list(self.backend_ids),
            },
            "seeds": list(self.seeds),
            "population_sizes": list(self.population_sizes),
            "termination_categories": list(self.termination_categories),
            "failure_policy": self.failure_policy,
            "reconstructable": self.reconstructable,
            "duplicate_tasks": self.duplicate_tasks,
            "output": self.output.as_dict(),
            "warnings": list(self.warnings),
            "errors": [error.as_dict() for error in self.errors],
            "next_actions": list(self.next_actions),
        }


def plan_study(spec: StudySpec, *, output: str | Path | None = None) -> StudyPlanReport:
    """Resolve and explain ``spec`` without creating a study or running a task."""
    if not isinstance(spec, StudySpec):
        from .errors import InvalidStudySpecError

        raise InvalidStudySpecError(
            operation="plan study",
            reason="INVALID_STUDY_SPEC",
            expected="validated StudySpec",
            actual=type(spec).__name__,
            action="Construct vamos.StudySpec(...) before calling plan_study.",
        )
    plan = resolve_spec(spec)
    destination = inspect_study_output(output)
    summaries = _summarize_plan(plan)
    errors = _output_errors(destination)
    status: PlanStatus = "ready" if not errors else "blocked"
    warnings: tuple[str, ...] = (
        ("Output availability is advisory and is not reserved; another process may occupy it after planning.",)
        if output is not None
        else ()
    )
    if summaries.reconstructable:
        warnings += ("Resolved built-ins are reconstructable; exact replayability is verified only after a run is published.",)
    next_actions = (
        (
            "Call vamos.create_study(spec, output=...) to publish this exact plan."
            if output is None
            else f"Call vamos.create_study(spec, output={os.fspath(output)!r}) to publish this exact plan."
        )
        if status == "ready"
        else "Choose an absent output path, then plan again before calling vamos.create_study(...).",
    )
    return StudyPlanReport(
        plan=plan,
        status=status,
        valid=True,
        total_evaluation_budget=summaries.total_evaluation_budget,
        problem_ids=summaries.problem_ids,
        algorithm_ids=summaries.algorithm_ids,
        operator_ids=summaries.operator_ids,
        backend_ids=summaries.backend_ids,
        seeds=summaries.seeds,
        population_sizes=summaries.population_sizes,
        termination_categories=summaries.termination_categories,
        failure_policy=spec.on_error,
        reconstructable=summaries.reconstructable,
        duplicate_tasks=False,
        output=destination,
        warnings=warnings,
        errors=errors,
        next_actions=next_actions,
    )


def inspect_study_output(output: str | Path | None) -> StudyPlanOutput:
    """Classify ``output`` with create-study collision semantics and no writes."""
    if output is None:
        return StudyPlanOutput(None, "not_checked", None, False, None)
    requested = os.fspath(output)
    destination = Path(output).absolute()
    advisory = "This check does not reserve the path; availability can change before creation."
    if not os.path.lexists(destination):
        return StudyPlanOutput(requested, "available", True, False, advisory)
    if destination.is_symlink():
        return StudyPlanOutput(requested, "existing_symlink", False, True, advisory)
    if destination.is_file():
        return StudyPlanOutput(requested, "existing_file", False, True, advisory)
    if destination.is_dir():
        try:
            if next(destination.iterdir(), None) is None:
                return StudyPlanOutput(requested, "empty_directory", False, True, advisory)
        except OSError:
            return StudyPlanOutput(requested, "unrelated_directory", False, True, advisory)
        marker = destination / "study-manifest.json"
        if os.path.lexists(marker):
            try:
                load_study(destination)
            except StudyError:
                return StudyPlanOutput(requested, "invalid_study_directory", False, True, advisory)
            return StudyPlanOutput(requested, "canonical_study", False, True, advisory)
        return StudyPlanOutput(requested, "unrelated_directory", False, True, advisory)
    return StudyPlanOutput(requested, "existing_other", False, True, advisory)


@dataclass(frozen=True, slots=True)
class _PlanSummaries:
    total_evaluation_budget: int
    problem_ids: tuple[str, ...]
    algorithm_ids: tuple[str, ...]
    operator_ids: tuple[str, ...]
    backend_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    population_sizes: tuple[int, ...]
    termination_categories: tuple[str, ...]
    reconstructable: bool


def _summarize_plan(plan: ResolvedStudyPlan) -> _PlanSummaries:
    problems: set[str] = set()
    algorithms: set[str] = set()
    operators: set[str] = set()
    backends: set[str] = set()
    seeds: set[int] = set()
    populations: set[int] = set()
    terminations: set[str] = set()
    total_budget = 0
    reconstructable = True
    for task in plan.tasks:
        resolved = task.resolved_run_spec
        problems.add(_component_id(resolved, "problem"))
        algorithms.add(_component_id(resolved, "algorithm"))
        operator_block = _mapping(resolved.get("operators"), "operators")
        operators.update(_component_id(operator_block, key) for key in sorted(operator_block))
        backend_block = _mapping(resolved.get("backend"), "backend")
        backends.update(_component_id(backend_block, key) for key in sorted(backend_block))
        termination = _mapping(resolved.get("termination"), "termination")
        terminations.add(_required_string(termination.get("component_id"), "termination.component_id"))
        termination_config = _mapping(termination.get("config"), "termination.config")
        total_budget += _required_int(termination_config.get("max_evaluations"), "termination.config.max_evaluations")
        seeds.add(_required_int(resolved.get("seed"), "seed"))
        population = _mapping(resolved.get("population"), "population")
        populations.add(_required_int(population.get("initial_size"), "population.initial_size"))
        reconstructable = reconstructable and _providers_are_built_in(resolved)
    return _PlanSummaries(
        total_evaluation_budget=total_budget,
        problem_ids=tuple(sorted(problems)),
        algorithm_ids=tuple(sorted(algorithms)),
        operator_ids=tuple(sorted(operators)),
        backend_ids=tuple(sorted(backends)),
        seeds=tuple(sorted(seeds)),
        population_sizes=tuple(sorted(populations)),
        termination_categories=tuple(sorted(terminations)),
        reconstructable=reconstructable,
    )


def _component_id(container: Mapping[str, Any], key: str) -> str:
    component = _mapping(container.get(key), key)
    return _required_string(component.get("component_id"), f"{key}.component_id")


def _providers_are_built_in(resolved: Mapping[str, Any]) -> bool:
    components: list[Mapping[str, Any]] = [
        _mapping(resolved.get("problem"), "problem"),
        _mapping(resolved.get("algorithm"), "algorithm"),
        _mapping(resolved.get("termination"), "termination"),
    ]
    operators = _mapping(resolved.get("operators"), "operators")
    components.extend(_mapping(value, f"operators.{key}") for key, value in operators.items())
    backends = _mapping(resolved.get("backend"), "backend")
    components.extend(_mapping(value, f"backend.{key}") for key, value in backends.items())
    return all(_mapping(component.get("provider"), "provider").get("type") == "built_in" for component in components)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssertionError(f"canonical resolver omitted {field}")
    return cast(Mapping[str, Any], value)


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise AssertionError(f"canonical resolver emitted invalid {field}")
    return value


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssertionError(f"canonical resolver emitted invalid {field}")
    return value


def _output_errors(output: StudyPlanOutput) -> tuple[StudyPlanDiagnostic, ...]:
    if not output.collision:
        return ()
    return (
        StudyPlanDiagnostic(
            operation="plan study",
            reason="OUTPUT_COLLISION",
            field="output",
            expected="destination path that does not exist",
            actual=output.status,
            action="Choose another output directory; VAMOS never overwrites, reuses, or merges studies.",
        ),
    )


__all__ = [
    "StudyPlanReport",
    "inspect_study_output",
    "plan_study",
]
