from __future__ import annotations

from dataclasses import dataclass

from .contracts import Credit, HierarchicalAction, Outcome, Regime, RegimeRouter, SearchState


@dataclass
class HeuristicRegimeRouter(RegimeRouter):
    """Deterministic regime routing using coarse search-state signals.

    Rules are intentionally explicit:
    - `REPAIR` when constraints are active and feasibility is poor.
    - `EXPAND` when budget is still early, diversity is low, or stagnation is high.
    - `REFINE` otherwise.
    """

    repair_feasible_ratio: float = 0.35
    repair_violation: float = 0.05
    early_budget_progress: float = 0.40
    low_diversity: float = 0.18
    high_stagnation: float = 0.60

    def route(self, search_state: SearchState) -> Regime:
        if search_state.is_constrained and (
            search_state.feasible_ratio < self.repair_feasible_ratio
            or search_state.mean_constraint_violation > self.repair_violation
        ):
            return Regime.REPAIR
        if (
            search_state.budget_progress < self.early_budget_progress
            or search_state.diversity < self.low_diversity
            or search_state.stagnation >= self.high_stagnation
        ):
            return Regime.EXPAND
        return Regime.REFINE

    def update(
        self,
        search_state: SearchState,
        action: HierarchicalAction,
        outcome: Outcome,
        credit: Credit,
    ) -> None:
        del search_state, action, outcome, credit


__all__ = ["HeuristicRegimeRouter"]
