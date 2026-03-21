from __future__ import annotations

from vamos.engine.adaptation.online_control import HierarchicalAction, OperatorFamily, Outcome, ParametricIntent, Regime, SearchState


def test_search_state_action_and_outcome_are_serializable() -> None:
    state = SearchState(
        host="nsgaii",
        step_index=3,
        generation=2,
        evaluations=24,
        population_size=12,
        objective_count=2,
        decision_dim=8,
        evaluation_budget=60,
        budget_progress=0.4,
        is_constrained=True,
        feasible_ratio=0.5,
        mean_constraint_violation=0.12,
        extent=0.21,
        diversity=0.18,
        stagnation=0.6,
        quality_indicator=-0.25,
        available_families=(OperatorFamily.SBX_LIKE, OperatorFamily.DE_LIKE),
        metadata={"nondominated_size": 5},
    )
    action = HierarchicalAction(
        regime=Regime.EXPAND,
        operator_family=OperatorFamily.DE_LIKE,
        parametric_intent=ParametricIntent(
            prototype="exploratory",
            exploration_strength=0.8,
            locality=0.2,
            mutation_strength=0.7,
            feasibility_bias=0.3,
        ),
    )
    outcome = Outcome(
        success=True,
        reward_hint=0.6,
        survivor_ratio=0.5,
        accepted_ratio=0.5,
        nd_insertions_ratio=0.25,
        feasible_ratio=0.7,
        feasible_delta=0.2,
        extent=0.26,
        extent_gain=0.05,
        diversity=0.22,
        diversity_delta=0.04,
        diversity_gain=0.04,
        quality_indicator=-0.20,
        quality_delta=0.05,
        overhead_ms=0.8,
    )

    state_dict = state.as_dict()
    action_dict = action.as_dict()
    outcome_dict = outcome.as_dict()

    assert state_dict["host"] == "nsgaii"
    assert state_dict["budget_progress"] == 0.4
    assert state_dict["extent"] == 0.21
    assert state_dict["available_families"] == ["sbx_like", "de_like"]
    assert action_dict["regime"] == "expand"
    assert action_dict["operator_family"] == "de_like"
    assert action_dict["parametric_intent"]["prototype"] == "exploratory"
    assert action_dict["parametric_intent"]["exploration_strength"] == 0.8
    assert action.parametric_intent.locality == 0.2
    assert outcome_dict["survivor_ratio"] == 0.5
    assert outcome_dict["nd_insertions_ratio"] == 0.25
    assert outcome_dict["extent_gain"] == 0.05
    assert outcome_dict["overhead_ms"] == 0.8
    assert outcome_dict["quality_delta"] == 0.05
