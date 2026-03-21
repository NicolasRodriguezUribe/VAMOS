from __future__ import annotations

from vamos.engine.adaptation.online_control import (
    CostAwareCreditModel,
    HierarchicalAction,
    OperatorFamily,
    Outcome,
    Regime,
    SearchState,
    SimpleImprovementCreditModel,
    build_intent_prototype,
)


def _state() -> SearchState:
    return SearchState(
        host="nsgaii",
        step_index=0,
        generation=0,
        evaluations=10,
        population_size=10,
        objective_count=2,
        decision_dim=4,
        evaluation_budget=40,
        budget_progress=0.25,
        extent=0.2,
        diversity=0.25,
        stagnation=0.1,
        quality_indicator=-0.2,
        available_families=(OperatorFamily.SBX_LIKE, OperatorFamily.DE_LIKE),
    )


def _action() -> HierarchicalAction:
    return HierarchicalAction(
        regime=Regime.EXPAND,
        operator_family=OperatorFamily.DE_LIKE,
        parametric_intent=build_intent_prototype("exploratory"),
    )


def test_simple_improvement_credit_is_bounded_and_uses_available_components() -> None:
    model = SimpleImprovementCreditModel()
    credit = model.compute(
        _state(),
        _action(),
        Outcome(
            success=True,
            reward_hint=0.1,
            accepted_ratio=0.6,
            nd_insertions_ratio=0.4,
            feasible_delta=0.2,
            extent_gain=0.1,
            diversity_gain=0.05,
        ),
    )

    assert 0.0 <= credit.reward <= 1.0
    assert credit.metadata["model"] == "simple_improvement"
    assert credit.metadata["components"]["accepted_ratio"] == 0.6


def test_simple_improvement_credit_degrades_gracefully_to_reward_hint() -> None:
    model = SimpleImprovementCreditModel()
    credit = model.compute(
        _state(),
        _action(),
        Outcome(success=False, reward_hint=0.3),
    )

    assert credit.reward == 0.3


def test_cost_aware_credit_penalizes_overhead_gently() -> None:
    simple = SimpleImprovementCreditModel()
    cost_aware = CostAwareCreditModel(overhead_scale_ms=1.0, max_penalty=0.2)
    outcome = Outcome(
        success=True,
        accepted_ratio=0.7,
        nd_insertions_ratio=0.5,
        extent_gain=0.1,
        diversity_gain=0.1,
        overhead_ms=5.0,
    )

    simple_credit = simple.compute(_state(), _action(), outcome)
    cost_credit = cost_aware.compute(_state(), _action(), outcome)

    assert 0.0 <= cost_credit.reward <= 1.0
    assert cost_credit.reward < simple_credit.reward
    assert cost_credit.metadata["model"] == "cost_aware"
    assert cost_credit.metadata["overhead_penalty"] > 0.0
