from __future__ import annotations

from vamos.engine.adaptation.online_control import AdaptiveHierarchicalJointPolicy, Credit, OperatorFamily, Regime, SearchState


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
        budget_progress=0.2,
        extent=0.15,
        diversity=0.15,
        stagnation=0.1,
        quality_indicator=-0.2,
        available_families=(OperatorFamily.SBX_LIKE, OperatorFamily.DE_LIKE),
    )


def test_adaptive_policy_state_roundtrip_preserves_tracker_state() -> None:
    policy = AdaptiveHierarchicalJointPolicy()
    state = _state()
    action = policy.select_action(state, Regime.EXPAND)
    policy.update(state, action, outcome=None, credit=Credit(reward=0.8, bounded_reward=0.8))

    exported = policy.export_state()
    restored = AdaptiveHierarchicalJointPolicy()
    restored.load_state(exported)

    assert restored.export_state() == exported
    next_action = restored.select_action(state, Regime.EXPAND)
    assert next_action.operator_family == action.operator_family
    assert next_action.parametric_intent.prototype == action.parametric_intent.prototype
