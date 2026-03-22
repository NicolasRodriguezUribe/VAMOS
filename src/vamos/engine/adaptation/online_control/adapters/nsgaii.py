from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from vamos.engine.variation import VariationPipeline
from vamos.engine.variation.protocol import CrossoverName, MutationName

from ..contracts import HierarchicalAction, HostAdapter, Outcome, SearchState
from .base import (
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


class NSGAIIOnlineControlAdapter(HostAdapter):
    host_name = "nsgaii"

    def build_search_state(self, host_state: Any) -> SearchState:
        X = np.asarray(host_state.X)
        F = np.asarray(host_state.F)
        G = getattr(host_state, "G", None)
        is_constrained, feasible_ratio, mean_violation = summarize_constraints(G)
        quality = float(getattr(host_state, "quality_indicator", scalar_quality_indicator(F, G)))
        families = tuple(getattr(host_state, "online_control_families", available_real_families()))
        return SearchState(
            host=self.host_name,
            step_index=int(getattr(host_state, "step", 0)),
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
            diversity=normalized_population_diversity(X, host_state.xl, host_state.xu),
            stagnation=normalized_stagnation(int(getattr(host_state, "stagnant_steps", 0))),
            quality_indicator=quality,
            available_families=families,
            metadata={
                "incremental_mode": bool(getattr(host_state, "incremental_mode", False)),
                "nondominated_size": nondominated_size(F, G),
                "variation": {
                    "cross_method": str(getattr(getattr(host_state, "variation", None), "cross_method", "")),
                    "mut_method": str(getattr(getattr(host_state, "variation", None), "mut_method", "")),
                },
            },
        )

    def decode_action(self, action: HierarchicalAction, host_state: Any) -> Any:
        descriptor = semantic_variation_descriptor(
            action,
            n_var=int(host_state.X.shape[1]),
            metadata={"host": self.host_name},
        )
        host_state.pending_online_descriptor = descriptor
        host_state.current_online_descriptor = descriptor
        return VariationPipeline(
            encoding=host_state.encoding,
            cross_method=cast(CrossoverName, descriptor.cross_method),
            cross_params=dict(descriptor.cross_params),
            mut_method=cast(MutationName, descriptor.mut_method),
            mut_params=dict(descriptor.mut_params),
            xl=host_state.xl,
            xu=host_state.xu,
            workspace=host_state.variation_workspace,
            repair_cfg=getattr(host_state, "repair_cfg", "auto"),
            problem=host_state.problem,
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
        survived_offspring = int(payload.get("survived_offspring", 0))
        offspring_count = max(1, int(payload.get("offspring_count", 0)))
        survivor_ratio = survived_offspring / offspring_count
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
        nd_insertions_ratio = clamp01(max(0.0, float(delta_nd or 0)) / float(offspring_count))
        accepted_ratio = survivor_ratio
        overhead_ms = payload.get("overhead_ms")

        reward = clamp01(
            0.35 * accepted_ratio
            + 0.20 * nd_insertions_ratio
            + 0.15 * max(0.0, quality_delta)
            + 0.15 * max(0.0, feasible_delta or 0.0)
            + 0.10 * max(0.0, extent_gain)
            + 0.05 * max(0.0, diversity_gain)
        )
        descriptor = getattr(host_state, "pending_online_descriptor", None)
        return Outcome(
            success=survived_offspring > 0 or reward > 0.0,
            reward_hint=reward,
            survivor_ratio=survivor_ratio,
            accepted_ratio=accepted_ratio,
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
                "survived_offspring": survived_offspring,
                "offspring_count": offspring_count,
                "delta_nondominated": delta_nd,
                "decoded_variation": descriptor.as_dict() if descriptor is not None else None,
            },
        )
__all__ = ["NSGAIIOnlineControlAdapter"]
