"""Deterministic, execution-free StudySpec resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

from vamos.experiment.artifacts.models import deep_freeze, deep_thaw
from vamos.experiment.artifacts.specs import RunSpecInputs, build_run_specs
from vamos.experiment.auto import _compute_max_evaluations, _compute_pop_size, _resolve_problem, _select_algorithm
from vamos.experiment.optimize import _build_algorithm_config
from vamos.experiment.runtime.catalog import resolve_engine_details
from vamos.foundation.encoding import normalize_encoding

from .errors import DuplicateStudyTaskError, StudyError, UnresolvedStudyTaskError
from .identity import compute_plan_id, compute_task_digest, compute_task_id
from .models import SCHEMA_VERSION, PlanTask, ResolvedStudyPlan, StudySpec
from .serialization import document_self_hash


@dataclass(frozen=True, slots=True)
class _Candidate:
    index: int
    problem: str
    algorithm: str
    seed: int


def resolve_spec(spec: StudySpec) -> ResolvedStudyPlan:
    """Resolve every matrix candidate and freeze one immutable sorted plan."""
    candidates = (
        _Candidate(index, problem, algorithm, seed)
        for index, (problem, algorithm, seed) in enumerate(product(spec.problems, spec.algorithms, spec.seeds))
    )
    tasks = tuple(_resolve_candidate(spec, candidate) for candidate in candidates)
    _reject_duplicates(tasks)
    sorted_tasks = tuple(sorted(tasks, key=lambda task: task.task_id))
    plan_id = compute_plan_id(task.task_id for task in sorted_tasks)
    provisional = ResolvedStudyPlan(plan_id=plan_id, tasks=sorted_tasks, document_sha256="")
    return ResolvedStudyPlan(plan_id=plan_id, tasks=sorted_tasks, document_sha256=_plan_hash(provisional))


def _resolve_candidate(spec: StudySpec, candidate: _Candidate) -> PlanTask:
    try:
        problem = _resolve_problem(
            candidate.problem,
            n_var=spec.n_var,
            n_obj=spec.n_obj,
            problem_kwargs=deep_thaw(spec.problem_kwargs),
        )
        n_var = int(problem.n_var)
        n_obj = int(problem.n_obj)
        encoding = normalize_encoding(getattr(problem, "encoding", "real"))
        algorithm = _select_algorithm(n_obj, encoding) if candidate.algorithm == "auto" else candidate.algorithm
        population_size = spec.pop_size or _compute_pop_size(n_var, n_obj)
        budget = spec.max_evaluations or _compute_max_evaluations(n_var, n_obj)
        engine, engine_source = resolve_engine_details(spec.engine, algorithm=algorithm)
        config = _resolved_algorithm_config(spec, candidate.algorithm, algorithm, population_size, n_var, n_obj, encoding)
        requested, resolved = build_run_specs(
            RunSpecInputs(
                problem_built_in=problem.__class__.__module__.startswith("vamos."),
                problem_label=candidate.problem,
                problem_kwargs=deep_thaw(spec.problem_kwargs),
                n_var_requested=spec.n_var,
                n_obj_requested=spec.n_obj,
                n_var=n_var,
                n_obj=n_obj,
                encoding=encoding,
                algorithm_requested=candidate.algorithm,
                algorithm=algorithm,
                algorithm_config=config,
                algorithm_config_explicit=bool(_algorithm_overrides(spec, candidate.algorithm, algorithm)),
                max_evaluations_requested=spec.max_evaluations,
                termination=("max_evaluations", budget),
                pop_size_requested=spec.pop_size,
                resolved_pop_size=config.get("pop_size", population_size),
                engine_requested=spec.engine,
                engine=engine,
                eval_strategy=spec.eval_strategy,
                seed_requested=candidate.seed,
                seed=candidate.seed,
                default_sources={
                    "algorithm": "auto" if candidate.algorithm == "auto" else "explicit",
                    "pop_size": "explicit" if spec.pop_size is not None else "auto",
                    "max_evaluations": "explicit" if spec.max_evaluations is not None else "auto",
                    "engine": engine_source,
                    "algorithm_config": "explicit" if _algorithm_overrides(spec, candidate.algorithm, algorithm) else "auto",
                },
            )
        )
    except StudyError:
        raise
    except Exception as exc:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="UNRESOLVED_TASK",
            field=f"$.matrix[{candidate.index}]",
            expected="supported problem, algorithm, backend, and complete JSON configuration",
            actual={
                "problem": candidate.problem,
                "algorithm": candidate.algorithm,
                "seed": candidate.seed,
                "error": type(exc).__name__,
            },
            action="Correct this matrix candidate; no study directory was published.",
        ) from exc
    task_id = compute_task_id(resolved)
    digest = compute_task_digest(resolved)
    return PlanTask(
        plan_index=candidate.index,
        requested_run=deep_freeze(requested),
        resolved_run_spec=deep_freeze(resolved),
        task_id=task_id,
        task_spec_sha256=digest,
    )


def _resolved_algorithm_config(
    spec: StudySpec,
    requested_algorithm: str,
    algorithm: str,
    population_size: int,
    n_var: int,
    n_obj: int,
    encoding: str,
) -> dict[str, Any]:
    config_pop_size: int | None = population_size
    if spec.pop_size is None and algorithm.lower() in {"moead", "nsgaiii", "rvea"}:
        config_pop_size = None
    built = _build_algorithm_config(
        algorithm,
        pop_size=config_pop_size,
        n_var=n_var,
        n_obj=n_obj,
        encoding=encoding,
    )
    config = dict(built.to_dict())
    overrides = _algorithm_overrides(spec, requested_algorithm, algorithm)
    unknown = sorted(set(overrides) - set(config))
    if unknown:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="UNRESOLVED_TASK",
            field=f"$.algorithm_configs.{algorithm}",
            expected=f"known configuration fields: {sorted(config)}",
            actual={"unknown": unknown},
            action="Remove unknown algorithm configuration fields; no study directory was published.",
        )
    config.update(overrides)
    return config


def _algorithm_overrides(spec: StudySpec, requested: str, resolved: str) -> dict[str, Any]:
    configs = spec.algorithm_configs or {}
    raw = configs.get(requested, configs.get(resolved, {}))
    if not isinstance(raw, Mapping):
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="UNRESOLVED_TASK",
            field=f"$.algorithm_configs.{requested}",
            expected="JSON object",
            actual=type(raw).__name__,
            action="Use an object keyed by supported algorithm configuration fields.",
        )
    thawed = deep_thaw(raw)
    if not isinstance(thawed, dict):
        raise AssertionError("algorithm override is not an object")
    return thawed


def _reject_duplicates(tasks: tuple[PlanTask, ...]) -> None:
    seen: set[str] = set()
    for task in tasks:
        if task.task_id in seen:
            raise DuplicateStudyTaskError(
                operation="resolve study plan",
                reason="DUPLICATE_CANONICAL_TASK",
                entity_id=task.task_id,
                expected="one matrix candidate per scientific task",
                actual="duplicate resolved run specification",
                action="Remove duplicate matrix values; no study directory was published.",
            )
        seen.add(task.task_id)


def _plan_hash(plan: ResolvedStudyPlan) -> str:
    tasks = [
        {
            "plan_index": task.plan_index,
            "requested_run": deep_thaw(task.requested_run),
            "resolved_run_spec": deep_thaw(task.resolved_run_spec),
            "task_id": task.task_id,
            "task_spec_sha256": task.task_spec_sha256,
        }
        for task in plan.tasks
    ]
    return document_self_hash(
        {
            "document_type": "vamos.resolved-study-plan",
            "schema_version": SCHEMA_VERSION,
            "plan_id": plan.plan_id,
            "task_count": len(tasks),
            "tasks": tasks,
            "integrity": {},
        }
    )


__all__ = ["resolve_spec"]
