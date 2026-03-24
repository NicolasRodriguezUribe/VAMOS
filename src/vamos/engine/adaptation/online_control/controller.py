from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import Credit, CreditModel, HierarchicalAction, HierarchicalPolicy, OperatorFamily, Outcome, Regime, RegimeRouter, SearchState
from .credit import CostAwareCreditModel, NoOpCreditModel, SimpleImprovementCreditModel
from .policies import (
    AdaptiveFlatOperatorPolicy,
    AdaptiveFlatParameterPolicy,
    AdaptiveHierarchicalJointPolicy,
    FlatOperatorPolicy,
    FlatParameterPolicy,
    HierarchicalJointPolicy,
)
from .prototypes import normalize_prototype_set
from .routers import HeuristicRegimeRouter, StaticRegimeRouter
from .storage import InMemoryTraceStore, TraceRow

ONLINE_CONTROL_ALLOWED_KEYS = {
    "enabled",
    "router",
    "policy",
    "credit_model",
    "trace_level",
    "fixed_family",
    "prototype_set",
    "policy_state",
}
_TRACE_LEVELS = {"basic", "off"}
_ROUTERS = {"heuristic", "static_expand", "static_refine", "static_repair"}
_POLICIES = {
    "flat_operator",
    "flat_parameter",
    "hierarchical_joint",
    "adaptive_flat_operator",
    "adaptive_flat_parameter",
    "adaptive_hierarchical_joint",
}
_CREDIT_MODELS = {"noop", "simple_improvement", "cost_aware"}


def _normalize_family(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    allowed = {family.value for family in OperatorFamily}
    if raw not in allowed:
        names = ", ".join(sorted(allowed))
        raise ValueError(f"online_control.fixed_family must be one of: {names}")
    return raw


def normalize_online_control_config(config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise TypeError("online_control must be a mapping.")

    normalized = {str(key): value for key, value in config.items()}
    unknown = sorted(set(normalized) - ONLINE_CONTROL_ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"Unknown online_control keys: {', '.join(unknown)}")

    enabled = normalized.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("online_control.enabled must be a boolean.")

    router = str(normalized.get("router", "heuristic")).strip().lower()
    policy = str(normalized.get("policy", "hierarchical_joint")).strip().lower()
    credit_model = str(normalized.get("credit_model", "simple_improvement")).strip().lower()
    trace_level = str(normalized.get("trace_level", "basic")).strip().lower()
    fixed_family = _normalize_family(normalized.get("fixed_family"))
    prototype_set = normalize_prototype_set(str(normalized.get("prototype_set", "default")))
    policy_state = normalized.get("policy_state")
    if policy_state is not None and not isinstance(policy_state, Mapping):
        raise TypeError("online_control.policy_state must be a mapping when provided.")

    if router not in _ROUTERS:
        names = ", ".join(sorted(_ROUTERS))
        raise ValueError(f"online_control.router must be one of: {names}")
    if policy not in _POLICIES:
        names = ", ".join(sorted(_POLICIES))
        raise ValueError(f"online_control.policy must be one of: {names}")
    if credit_model not in _CREDIT_MODELS:
        names = ", ".join(sorted(_CREDIT_MODELS))
        raise ValueError(f"online_control.credit_model must be one of: {names}")
    if trace_level not in _TRACE_LEVELS:
        raise ValueError("online_control.trace_level must be 'basic' or 'off'.")

    payload = {
        "enabled": enabled,
        "router": router,
        "policy": policy,
        "credit_model": credit_model,
        "trace_level": trace_level,
        "prototype_set": prototype_set,
    }
    if fixed_family is not None:
        payload["fixed_family"] = fixed_family
    if policy_state is not None:
        payload["policy_state"] = dict(policy_state)
    return payload


def _build_router(name: str) -> RegimeRouter:
    if name == "heuristic":
        return HeuristicRegimeRouter()
    if name == "static_expand":
        return StaticRegimeRouter(fixed_regime=Regime.EXPAND)
    if name == "static_refine":
        return StaticRegimeRouter(fixed_regime=Regime.REFINE)
    if name == "static_repair":
        return StaticRegimeRouter(fixed_regime=Regime.REPAIR)
    raise ValueError(f"Unsupported online_control router '{name}'.")


def _build_policy(
    name: str,
    *,
    fixed_family: str | None,
    prototype_set: str,
    policy_state: Mapping[str, Any] | None = None,
) -> HierarchicalPolicy:
    policy: HierarchicalPolicy
    if name == "flat_operator":
        policy = FlatOperatorPolicy(prototype_set=prototype_set)
    elif name == "flat_parameter":
        family = OperatorFamily(fixed_family or OperatorFamily.SBX_LIKE.value)
        policy = FlatParameterPolicy(fixed_family=family, prototype_set=prototype_set)
    elif name == "hierarchical_joint":
        policy = HierarchicalJointPolicy(prototype_set=prototype_set)
    elif name == "adaptive_flat_operator":
        policy = AdaptiveFlatOperatorPolicy(prototype_set=prototype_set)
    elif name == "adaptive_flat_parameter":
        family = OperatorFamily(fixed_family or OperatorFamily.SBX_LIKE.value)
        policy = AdaptiveFlatParameterPolicy(fixed_family=family, prototype_set=prototype_set)
    elif name == "adaptive_hierarchical_joint":
        family = OperatorFamily(fixed_family) if fixed_family is not None else None
        policy = AdaptiveHierarchicalJointPolicy(fixed_family=family, prototype_set=prototype_set)
    else:
        raise ValueError(f"Unsupported online_control policy '{name}'.")
    if policy_state is not None:
        load_state = getattr(policy, "load_state", None)
        if not callable(load_state):
            raise ValueError(f"online_control policy '{name}' does not support policy_state import.")
        load_state(policy_state)
    return policy


def _build_credit_model(name: str) -> CreditModel:
    if name == "noop":
        return NoOpCreditModel()
    if name == "simple_improvement":
        return SimpleImprovementCreditModel()
    if name == "cost_aware":
        return CostAwareCreditModel()
    raise ValueError(f"Unsupported online_control credit model '{name}'.")


@dataclass
class _PendingStep:
    search_state: SearchState
    action: HierarchicalAction | None = None


class OnlineControlController:
    def __init__(
        self,
        *,
        router: RegimeRouter | None = None,
        policy: HierarchicalPolicy | None = None,
        credit_model: CreditModel | None = None,
        trace_store: InMemoryTraceStore | None = None,
        trace_level: str = "basic",
    ) -> None:
        self.router = router or HeuristicRegimeRouter()
        self.policy = policy or HierarchicalJointPolicy()
        self.credit_model = credit_model or SimpleImprovementCreditModel()
        self.trace_level = trace_level
        self.trace_store = trace_store or InMemoryTraceStore(enabled=trace_level != "off")
        self._pending: _PendingStep | None = None
        self._profile_totals: dict[str, float] = {
            "start_step_time_ms": 0.0,
            "router_time_ms": 0.0,
            "policy_select_time_ms": 0.0,
            "policy_update_time_ms": 0.0,
            "trace_time_ms": 0.0,
            "controller_time_ms": 0.0,
        }

    def _record_profile(self, key: str, elapsed_ms: float) -> None:
        self._profile_totals[key] = self._profile_totals.get(key, 0.0) + float(elapsed_ms)

    def start_step(self, search_state: SearchState) -> None:
        t0 = time.perf_counter()
        if self._pending is not None:
            raise RuntimeError("online_control step already started; finalize the previous step first.")
        self._pending = _PendingStep(search_state=search_state)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._record_profile("start_step_time_ms", elapsed_ms)
        self._record_profile("controller_time_ms", elapsed_ms)

    def select_action(self) -> HierarchicalAction:
        if self._pending is None:
            raise RuntimeError("online_control step not started.")
        router_t0 = time.perf_counter()
        regime = self.router.route(self._pending.search_state)
        router_ms = (time.perf_counter() - router_t0) * 1000.0
        self._record_profile("router_time_ms", router_ms)
        policy_t0 = time.perf_counter()
        action = self.policy.select_action(self._pending.search_state, regime)
        policy_ms = (time.perf_counter() - policy_t0) * 1000.0
        self._record_profile("policy_select_time_ms", policy_ms)
        self._record_profile("controller_time_ms", router_ms + policy_ms)
        self._pending.action = action
        return action

    def finalize_step(self, outcome: Outcome) -> Credit:
        if self._pending is None or self._pending.action is None:
            raise RuntimeError("online_control step must be started and selected before finalize_step().")

        search_state = self._pending.search_state
        action = self._pending.action
        update_t0 = time.perf_counter()
        credit = self.credit_model.compute(search_state, action, outcome)
        self.policy.update(search_state, action, outcome, credit)
        self.router.update(search_state, action, outcome, credit)
        update_ms = (time.perf_counter() - update_t0) * 1000.0
        self._record_profile("policy_update_time_ms", update_ms)
        trace_t0 = time.perf_counter()
        self.trace_store.append(
            TraceRow(
                step_index=search_state.step_index,
                search_state=search_state,
                action=action,
                outcome=outcome,
                credit=credit,
            )
        )
        trace_ms = (time.perf_counter() - trace_t0) * 1000.0
        self._record_profile("trace_time_ms", trace_ms)
        self._record_profile("controller_time_ms", update_ms + trace_ms)
        self._pending = None
        return credit

    def trace_rows(self) -> list[TraceRow]:
        return self.trace_store.rows()

    def trace_dicts(self) -> list[dict[str, object]]:
        return self.trace_store.to_dicts()

    def trace_flat_dicts(self) -> list[dict[str, object]]:
        return self.trace_store.to_flat_dicts()

    def summary_rows(self) -> list[dict[str, object]]:
        return self.trace_store.summary_rows()

    def run_summary(self) -> dict[str, object]:
        summary = dict(self.trace_store.run_summary())
        summary["runtime_profile"] = self.profiling_summary()
        summary.update(self.profiling_summary())
        return summary

    def profiling_summary(self) -> dict[str, float]:
        return {key: float(value) for key, value in self._profile_totals.items()}

    def result_payload(self) -> dict[str, object]:
        return {
            "enabled": True,
            "trace_rows": self.trace_flat_dicts(),
            "trace": self.trace_dicts(),
            "summary": self.summary_rows(),
            "run_summary": self.run_summary(),
            "policy_state": self.policy_state(),
            "controller_profile": self.profiling_summary(),
        }

    def policy_state(self) -> dict[str, object] | None:
        export_state = getattr(self.policy, "export_state", None)
        if not callable(export_state):
            return None
        payload = export_state()
        return payload if isinstance(payload, dict) else None


def build_online_control_controller(config: Mapping[str, Any] | None) -> OnlineControlController | None:
    normalized = normalize_online_control_config(config)
    if normalized is None or not normalized["enabled"]:
        return None
    return OnlineControlController(
        router=_build_router(str(normalized["router"])),
        policy=_build_policy(
            str(normalized["policy"]),
            fixed_family=normalized.get("fixed_family"),
            prototype_set=str(normalized["prototype_set"]),
            policy_state=normalized.get("policy_state"),
        ),
        credit_model=_build_credit_model(str(normalized["credit_model"])),
        trace_level=str(normalized["trace_level"]),
    )


__all__ = [
    "ONLINE_CONTROL_ALLOWED_KEYS",
    "OnlineControlController",
    "build_online_control_controller",
    "normalize_online_control_config",
]
