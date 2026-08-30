"""Exact built-in replay-plan reconstruction from a verified resolved spec."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from vamos.engine.algorithm.config import (
    AGEMOEAConfig,
    IBEAConfig,
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    RVEAConfig,
    SMPSOConfig,
    SMSEMOAConfig,
    SPEA2Config,
)
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol, EngineName
from vamos.experiment.types import TerminationSpec
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.problem.registry import make_problem_selection

from .component_support import component_reconstructability
from .errors import (
    ComponentNotReconstructableError,
    OutputCollisionError,
    ReplayUnavailableError,
    ResolvedSpecMismatchError,
    UnsupportedReplayProviderError,
)
from .jsonio import canonical_json_bytes, sha256_bytes
from .lineage import MAX_REPLAY_LINEAGE_DEPTH
from .models import RunManifest, StoredRun, deep_freeze, deep_thaw
from .reports import VerificationReport
from .specs import RunSpecInputs, build_run_specs

_CONFIG_TYPES = {
    "agemoea": AGEMOEAConfig,
    "ibea": IBEAConfig,
    "moead": MOEADConfig,
    "nsgaii": NSGAIIConfig,
    "nsgaiii": NSGAIIIConfig,
    "rvea": RVEAConfig,
    "smpso": SMPSOConfig,
    "smsemoa": SMSEMOAConfig,
    "spea2": SPEA2Config,
}
_OPERATOR_INPUT_NAMES = {
    "differential_evolution": "de",
    "order": "ox",
    "polynomial": "pm",
}
_OPERATOR_ROLES = ("crossover", "initializer", "mutation", "repair", "selection")


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    """Internal immutable, pre-validated execution plan."""

    source_root: Path
    output_root: Path
    source_run_id: str
    root_run_id: str
    lineage_depth: int
    new_run_id: str
    source_manifest_sha256: str
    requested_spec: Mapping[str, Any]
    resolved_spec: Mapping[str, Any]
    problem: str
    n_var: int
    n_obj: int
    encoding: str
    algorithm: str
    algorithm_config: AlgorithmConfigProtocol
    termination: TerminationSpec
    engine: EngineName
    seed: int
    expected_arrays: tuple[str, ...]
    replay_plan_sha256: str


def build_replay_plan(stored: StoredRun, verification: VerificationReport, output: str | Path | None) -> ReplayPlan:
    """Construct and semantically prove a plan before any optimization begins."""
    _ensure_components(stored)
    if verification.effective_replayability != "exact":
        raise ReplayUnavailableError(
            operation="reproduce run",
            field="$.effective_replayability",
            path=stored.root,
            reason="is not exact",
            expected="exact",
            actual=verification.effective_replayability,
            action="Run verify_run(path, require_level='exact') and correct every blocking finding.",
        )
    resolved = stored.manifest.resolved_spec.as_dict()
    problem, n_var, n_obj, encoding = _problem_fields(resolved, stored.root)
    algorithm = _component_name(_mapping(resolved.get("algorithm")).get("component_id"), "vamos.algorithm")
    if algorithm is None or algorithm not in _CONFIG_TYPES:
        raise _component_error(stored.root, "$.resolved_spec.algorithm.component_id", algorithm)
    config = _reconstruct_algorithm_config(algorithm, resolved, stored.root)
    termination = _termination(resolved, stored.root)
    engine = _engine(resolved, stored.root)
    seed = _integer(resolved.get("seed"), stored.root, "$.resolved_spec.seed")
    _assert_problem_registered(problem, n_var, n_obj, encoding, stored.root)
    _assert_semantic_equality(resolved, problem, n_var, n_obj, encoding, algorithm, config, termination, engine, seed, stored.root)
    new_run_id = str(uuid.uuid4())
    output_root = _output_root(stored.root, new_run_id, output)
    _reject_collision(output_root)
    root_run_id, depth = _lineage(stored.manifest, stored.root)
    source_hash = str(stored.manifest["integrity"]["manifest_sha256"])
    descriptor = stored.manifest.artifact("result_bundle")
    expected_arrays = tuple(sorted(descriptor.array_contract)) if descriptor is not None and descriptor.array_contract else ()
    payload = {
        "source_run_id": stored.manifest.run_id,
        "source_manifest_sha256": source_hash,
        "resolved_spec": resolved,
        "expected_arrays": list(expected_arrays),
        "compatibility_level": verification.effective_replayability,
    }
    plan_hash = sha256_bytes(canonical_json_bytes(payload))
    return ReplayPlan(
        source_root=stored.root,
        output_root=output_root,
        source_run_id=stored.manifest.run_id,
        root_run_id=root_run_id,
        lineage_depth=depth,
        new_run_id=new_run_id,
        source_manifest_sha256=source_hash,
        requested_spec=cast(Mapping[str, Any], deep_freeze(stored.manifest.requested_spec)),
        resolved_spec=cast(Mapping[str, Any], deep_freeze(resolved)),
        problem=problem,
        n_var=n_var,
        n_obj=n_obj,
        encoding=encoding,
        algorithm=algorithm,
        algorithm_config=config,
        termination=termination,
        engine=cast(EngineName, engine),
        seed=seed,
        expected_arrays=expected_arrays,
        replay_plan_sha256=plan_hash,
    )


def _ensure_components(stored: StoredRun) -> None:
    status, reasons = component_reconstructability(stored.manifest)
    if status != "reconstructable":
        reason = reasons[0] if reasons else None
        error_type = (
            UnsupportedReplayProviderError
            if reason is not None and reason.code in {"custom_provider", "unsupported_provider"}
            else ComponentNotReconstructableError
        )
        raise error_type(
            operation="reproduce run",
            field=reason.field if reason is not None else "$.resolved_spec",
            path=stored.root,
            reason="contains a component outside exact built-in replay support",
            expected="reconstructable built-in components",
            actual=status,
            action=reason.action if reason is not None else "Regenerate with supported built-ins.",
        )


def _reconstruct_algorithm_config(algorithm: str, resolved: Mapping[str, Any], root: Path) -> AlgorithmConfigProtocol:
    algorithm_descriptor = _mapping(resolved.get("algorithm"))
    raw_config = algorithm_descriptor.get("config")
    if not isinstance(raw_config, Mapping):
        raise _component_error(root, "$.resolved_spec.algorithm.config", raw_config)
    config = cast(dict[str, object], deep_thaw(deep_freeze(raw_config)))
    operators = resolved.get("operators")
    if not isinstance(operators, Mapping):
        raise _component_error(root, "$.resolved_spec.operators", operators)
    for role, descriptor in operators.items():
        if role not in _OPERATOR_ROLES or not isinstance(descriptor, Mapping):
            raise _component_error(root, f"$.resolved_spec.operators.{role}", descriptor)
        _restore_operator(config, str(role), descriptor, root)
    config_type = _CONFIG_TYPES[algorithm]
    try:
        return cast(AlgorithmConfigProtocol, cast(Any, config_type).from_dict(config))
    except (TypeError, ValueError) as exc:
        raise ComponentNotReconstructableError(
            operation="reproduce run",
            field="$.resolved_spec.algorithm.config",
            path=root,
            reason="cannot reconstruct the typed built-in algorithm configuration",
            expected=config_type.__name__,
            actual=f"{type(exc).__name__}: {exc}",
            action="Regenerate the run with a complete current resolved configuration.",
        ) from exc


def _restore_operator(config: dict[str, object], role: str, descriptor: Mapping[str, Any], root: Path) -> None:
    resolution = descriptor.get("resolution")
    if isinstance(resolution, Mapping) and resolution.get("active") is False:
        if role == "repair":
            config[role] = "auto"
        return
    stable_name = _component_name(descriptor.get("component_id"), "vamos.operator")
    parameters = descriptor.get("config")
    if stable_name is None or not isinstance(parameters, Mapping):
        raise _component_error(root, f"$.resolved_spec.operators.{role}", descriptor)
    name = _OPERATOR_INPUT_NAMES.get(stable_name, stable_name)
    values = cast(dict[str, Any], deep_thaw(deep_freeze(parameters)))
    if role == "initializer":
        config[role] = {"type": name, **values}
    else:
        config[role] = (name, values)


def _assert_semantic_equality(
    source: Mapping[str, Any],
    problem: str,
    n_var: int,
    n_obj: int,
    encoding: str,
    algorithm: str,
    config: AlgorithmConfigProtocol,
    termination: TerminationSpec,
    engine: str,
    seed: int,
    root: Path,
) -> None:
    _, reconstructed = build_run_specs(
        RunSpecInputs(
            problem_built_in=True,
            problem_label=problem,
            problem_kwargs=None,
            n_var_requested=n_var,
            n_obj_requested=n_obj,
            n_var=n_var,
            n_obj=n_obj,
            encoding=encoding,
            algorithm_requested=algorithm,
            algorithm=algorithm,
            algorithm_config=dict(config.to_dict()),
            algorithm_config_explicit=True,
            max_evaluations_requested=None,
            termination=termination,
            pop_size_requested=None,
            resolved_pop_size=config.to_dict().get("pop_size"),
            engine_requested=engine,
            engine=engine,
            eval_strategy=None,
            seed_requested=seed,
            seed=seed,
            default_sources=_default_sources(source),
        )
    )
    expected_bytes = canonical_json_bytes(source)
    actual_bytes = canonical_json_bytes(reconstructed)
    if expected_bytes == actual_bytes:
        return
    raise ResolvedSpecMismatchError(
        operation="reproduce run",
        field="$.resolved_spec",
        path=root,
        reason="does not equal the explicitly reconstructed effective specification",
        expected=sha256_bytes(expected_bytes),
        actual=sha256_bytes(actual_bytes),
        action="Regenerate the source with the current complete canonical spec; current defaults were not substituted.",
    )


def _problem_fields(resolved: Mapping[str, Any], root: Path) -> tuple[str, int, int, str]:
    descriptor = _mapping(resolved.get("problem"))
    problem = _component_name(descriptor.get("component_id"), "vamos.problem")
    config = _mapping(descriptor.get("config"))
    if problem is None:
        raise _component_error(root, "$.resolved_spec.problem.component_id", descriptor.get("component_id"))
    n_var = _integer(config.get("n_var"), root, "$.resolved_spec.problem.config.n_var")
    n_obj = _integer(config.get("n_obj"), root, "$.resolved_spec.problem.config.n_obj")
    encoding = config.get("encoding")
    if not isinstance(encoding, str):
        raise _component_error(root, "$.resolved_spec.problem.config.encoding", encoding)
    extras = set(config) - {"constraint_convention", "encoding", "n_obj", "n_var"}
    if extras:
        raise _component_error(root, "$.resolved_spec.problem.config", {"unsupported_fields": sorted(extras)})
    return problem, n_var, n_obj, encoding


def _assert_problem_registered(problem: str, n_var: int, n_obj: int, encoding: str, root: Path) -> None:
    try:
        selection = make_problem_selection(problem, n_var=n_var, n_obj=n_obj)
    except (KeyError, TypeError, ValueError) as exc:
        raise _component_error(root, "$.resolved_spec.problem", problem) from exc
    registered_encoding = normalize_encoding(selection.spec.encoding)
    if registered_encoding != encoding:
        raise ResolvedSpecMismatchError(
            operation="reproduce run",
            field="$.resolved_spec.problem.config.encoding",
            path=root,
            reason="differs from the current built-in problem registration",
            expected=encoding,
            actual=registered_encoding,
            action="Use the exact VAMOS implementation that created the source run.",
        )


def _termination(resolved: Mapping[str, Any], root: Path) -> TerminationSpec:
    descriptor = _mapping(resolved.get("termination"))
    name = _component_name(descriptor.get("component_id"), "vamos.termination")
    config = _mapping(descriptor.get("config"))
    if name == "max_evaluations":
        value = _integer(config.get("max_evaluations"), root, "$.resolved_spec.termination.config.max_evaluations")
        if config.get("hard_max_evaluations") != value or set(config) != {"max_evaluations", "hard_max_evaluations"}:
            raise _component_error(root, "$.resolved_spec.termination.config", config)
        return (name, value)
    if name == "hv":
        return (name, cast(dict[str, object], deep_thaw(deep_freeze(config))))
    raise _component_error(root, "$.resolved_spec.termination.component_id", name)


def _engine(resolved: Mapping[str, Any], root: Path) -> str:
    backend = _mapping(resolved.get("backend"))
    kernel = _mapping(backend.get("kernel"))
    engine = _component_name(kernel.get("component_id"), "vamos.kernel")
    resolution = _mapping(kernel.get("resolution"))
    if engine not in {"numpy", "numba", "moocore"} or resolution.get("name") != engine:
        raise _component_error(root, "$.resolved_spec.backend.kernel", kernel)
    evaluation = _mapping(backend.get("evaluation"))
    evaluation_name = _component_name(evaluation.get("component_id"), "vamos.evaluation")
    if evaluation_name != "serial" or evaluation.get("config") != {"ordered": True, "workers": 1}:
        raise _component_error(root, "$.resolved_spec.backend.evaluation", evaluation)
    return engine


def _lineage(manifest: RunManifest, root_path: Path) -> tuple[str, int]:
    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping):
        return manifest.run_id, 1
    root = lineage.get("root_run_id")
    depth = lineage.get("depth")
    if isinstance(root, str) and isinstance(depth, int):
        if depth >= MAX_REPLAY_LINEAGE_DEPTH:
            raise ReplayUnavailableError(
                operation="reproduce run",
                field="$.lineage.depth",
                path=root_path,
                reason="has reached the bounded replay-lineage limit",
                expected=f"depth below {MAX_REPLAY_LINEAGE_DEPTH}",
                actual=depth,
                action="Use the root or an earlier replay as the source instead of extending this lineage.",
                optimization_executed=False,
            )
        return str(root), depth + 1
    return manifest.run_id, 1


def _output_root(source: Path, new_run_id: str, output: str | Path | None) -> Path:
    candidate = Path(output) if output is not None else source.parent / "replays" / new_run_id
    return candidate.absolute()


def _reject_collision(path: Path) -> None:
    if os.path.lexists(path):
        raise OutputCollisionError(
            operation="reproduce run",
            path=path,
            reason="already exists",
            expected="an output path that does not exist",
            actual="occupied output path",
            action="Choose another --output directory; replay never overwrites or merges runs.",
        )


def _default_sources(resolved: Mapping[str, Any]) -> dict[str, str]:
    items = resolved.get("defaults_applied")
    if not isinstance(items, list):
        return {}
    return {
        str(item["field"]).lstrip("/"): str(item["source"])
        for item in items
        if isinstance(item, Mapping) and "field" in item and "source" in item
    }


def _component_name(value: object, namespace: str) -> str | None:
    prefix = f"{namespace}:"
    if not isinstance(value, str) or not value.startswith(prefix) or not value.endswith("@1"):
        return None
    return value[len(prefix) : -2]


def _integer(value: object, root: Path, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _component_error(root, field, value)
    return value


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _component_error(root: Path, field: str, actual: object) -> ComponentNotReconstructableError:
    return ComponentNotReconstructableError(
        operation="reproduce run",
        field=field,
        path=root,
        reason="is not a complete supported built-in replay component",
        expected="stable schema-1 built-in descriptor",
        actual=actual,
        action="Regenerate the run with a current registered built-in component.",
    )


__all__ = ["ReplayPlan", "build_replay_plan"]
