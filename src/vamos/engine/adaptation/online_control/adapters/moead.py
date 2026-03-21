from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from vamos.engine.operators.policies.moead import build_variation_operators

from ..contracts import HierarchicalAction, HostAdapter, Outcome, SearchState
from .base import (
    DecodedMOEADVariation,
    available_real_families,
    budget_progress,
    clamp01,
    nondominated_size,
    normalized_objective_extent,
    normalized_population_diversity,
    normalized_stagnation,
    scalar_quality_indicator,
    semantic_variation_descriptor,
    summarize_constraints,
)


class MOEADOnlineControlAdapter(HostAdapter):
    host_name = "moead"

    def build_search_state(self, host_state: Any) -> SearchState:
        X = np.asarray(getattr(host_state, "X"))
        F = np.asarray(getattr(host_state, "F"))
        G = getattr(host_state, "G", None)
        is_constrained, feasible_ratio, mean_violation = summarize_constraints(G)
        quality = float(getattr(host_state, "quality_indicator", scalar_quality_indicator(F, G)))
        families = tuple(getattr(host_state, "online_control_families", available_real_families()))
        return SearchState(
            host=self.host_name,
            step_index=int(getattr(host_state, "generation", 0)),
            generation=int(getattr(host_state, "generation", 0)),
            evaluations=int(getattr(host_state, "n_eval", 0)),
            population_size=int(X.shape[0]),
            objective_count=int(F.shape[1]),
            decision_dim=int(X.shape[1]),
            evaluation_budget=int(getattr(host_state, "max_evaluations", 0)) or None,
            budget_progress=budget_progress(int(getattr(host_state, "n_eval", 0)), getattr(host_state, "max_evaluations", None)),
            is_constrained=is_constrained,
            feasible_ratio=feasible_ratio,
            mean_constraint_violation=mean_violation,
            extent=normalized_objective_extent(F),
            diversity=normalized_population_diversity(X, getattr(host_state, "xl"), getattr(host_state, "xu")),
            stagnation=normalized_stagnation(int(getattr(host_state, "stagnant_steps", 0))),
            quality_indicator=quality,
            available_families=families,
            metadata={
                "batch_size": int(getattr(host_state, "batch_size", 1)),
                "nondominated_size": nondominated_size(F, G),
                "current_cross_method": str(getattr(host_state, "current_cross_method", "")),
                "current_mutation_method": str(getattr(host_state, "current_mutation_method", "")),
            },
        )

    def decode_action(self, action: HierarchicalAction, host_state: Any) -> Any:
        descriptor = semantic_variation_descriptor(
            action,
            n_var=int(getattr(host_state, "X").shape[1]),
            metadata={"host": self.host_name},
        )
        cfg = dict(getattr(host_state, "base_variation_config", {}))
        cfg["crossover"] = (descriptor.cross_method, dict(descriptor.cross_params))
        cfg["mutation"] = (descriptor.mut_method, dict(descriptor.mut_params))
        crossover_fn, mutation_fn = build_variation_operators(
            cfg,
            getattr(host_state, "encoding"),
            int(getattr(host_state, "X").shape[1]),
            getattr(host_state, "xl"),
            getattr(host_state, "xu"),
            getattr(host_state, "rng"),
            mixed_spec=getattr(getattr(host_state, "problem", None), "mixed_spec", None),
        )
        host_state.pending_online_descriptor = descriptor
        host_state.current_online_descriptor = descriptor
        return DecodedMOEADVariation(
            descriptor=descriptor,
            crossover_fn=crossover_fn,
            mutation_fn=mutation_fn,
            cross_is_de=descriptor.cross_method == "de",
        )

    def build_outcome(
        self,
        *,
        before: SearchState,
        after: SearchState,
        action: HierarchicalAction,
        host_state: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> Outcome:
        payload = dict(metadata or {})
        replaced = int(payload.get("replaced", 0))
        batch_size = max(1, int(payload.get("batch_size", 1)))
        survivor_ratio = clamp01(replaced / batch_size)
        before_nd = before.metadata.get("nondominated_size")
        after_nd = after.metadata.get("nondominated_size")
        delta_nd = None
        if isinstance(before_nd, (int, np.integer)) and isinstance(after_nd, (int, np.integer)):
            delta_nd = int(after_nd) - int(before_nd)
        feasible_delta = after.feasible_ratio - before.feasible_ratio if before.is_constrained or after.is_constrained else None
        extent_gain = after.extent - before.extent
        diversity_delta = after.diversity - before.diversity
        diversity_gain = diversity_delta
        quality_delta = after.quality_indicator - before.quality_indicator
        nd_insertions_ratio = clamp01(max(0.0, float(delta_nd or 0)) / float(batch_size))
        overhead_ms = payload.get("overhead_ms")
        reward = clamp01(
            0.35 * survivor_ratio
            + 0.20 * nd_insertions_ratio
            + 0.25 * max(0.0, quality_delta)
            + 0.15 * max(0.0, feasible_delta or 0.0)
            + 0.10 * max(0.0, extent_gain)
            + 0.10 * max(0.0, diversity_gain)
        )
        descriptor = getattr(host_state, "pending_online_descriptor", None)
        return Outcome(
            success=replaced > 0 or reward > 0.0,
            reward_hint=reward,
            survivor_ratio=survivor_ratio,
            accepted_ratio=survivor_ratio,
            nd_insertions_ratio=nd_insertions_ratio,
            feasible_ratio=after.feasible_ratio if after.is_constrained else None,
            feasible_delta=feasible_delta,
            extent=after.extent,
            extent_gain=extent_gain,
            diversity=after.diversity,
            diversity_delta=diversity_delta,
            diversity_gain=diversity_gain,
            quality_indicator=after.quality_indicator,
            quality_delta=quality_delta,
            overhead_ms=float(overhead_ms) if overhead_ms is not None else None,
            metadata={
                "operator_family": action.operator_family.value,
                "replaced": replaced,
                "batch_size": batch_size,
                "delta_nondominated": delta_nd,
                "decoded_variation": descriptor.as_dict() if descriptor is not None else None,
            },
        )


__all__ = ["MOEADOnlineControlAdapter"]
