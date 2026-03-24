from __future__ import annotations

from vamos.engine.adaptation.online_control import (
    AdaptiveFlatOperatorPolicy,
    AdaptiveFlatParameterPolicy,
    AdaptiveHierarchicalJointPolicy,
    Credit,
    FlatOperatorPolicy,
    FlatParameterPolicy,
    HeuristicRegimeRouter,
    HierarchicalJointPolicy,
    OperatorFamily,
    Regime,
    SearchState,
    available_intent_prototypes,
    build_intent_prototype,
)


def _state(**overrides: object) -> SearchState:
    payload: dict[str, object] = {
        "host": "nsgaii",
        "step_index": 0,
        "generation": 0,
        "evaluations": 10,
        "population_size": 10,
        "objective_count": 2,
        "decision_dim": 4,
        "evaluation_budget": 40,
        "budget_progress": 0.25,
        "is_constrained": False,
        "feasible_ratio": 1.0,
        "mean_constraint_violation": 0.0,
        "extent": 0.25,
        "diversity": 0.3,
        "stagnation": 0.0,
        "quality_indicator": -0.2,
        "available_families": (OperatorFamily.SBX_LIKE, OperatorFamily.DE_LIKE),
    }
    payload.update(overrides)
    return SearchState(**payload)


def test_heuristic_regime_router_routes_repair_expand_and_refine() -> None:
    router = HeuristicRegimeRouter()

    repair_state = _state(is_constrained=True, feasible_ratio=0.2, mean_constraint_violation=0.2)
    expand_state = _state(budget_progress=0.2, diversity=0.15)
    refine_state = _state(budget_progress=0.8, diversity=0.35, stagnation=0.2)

    assert router.route(repair_state) is Regime.REPAIR
    assert router.route(expand_state) is Regime.EXPAND
    assert router.route(refine_state) is Regime.REFINE


def test_flat_operator_policy_changes_family_but_keeps_neutral_intent() -> None:
    policy = FlatOperatorPolicy()
    action = policy.select_action(_state(), Regime.EXPAND)

    assert action.operator_family is OperatorFamily.DE_LIKE
    assert action.parametric_intent.prototype == "balanced"
    assert action.parametric_intent.exploration_strength == 0.5
    assert action.parametric_intent.locality == 0.5
    assert action.parametric_intent.mutation_strength == 0.5


def test_flat_parameter_policy_keeps_fixed_family_and_adapts_intent() -> None:
    policy = FlatParameterPolicy(fixed_family=OperatorFamily.SBX_LIKE)
    action = policy.select_action(_state(diversity=0.1, stagnation=0.7), Regime.EXPAND)

    assert action.operator_family is OperatorFamily.SBX_LIKE
    assert action.parametric_intent.prototype == "mutation_heavy"
    assert action.parametric_intent.mutation_strength > 0.8


def test_hierarchical_joint_policy_conditions_family_on_regime() -> None:
    policy = HierarchicalJointPolicy()

    expand_action = policy.select_action(_state(budget_progress=0.2, diversity=0.12), Regime.EXPAND)
    refine_action = policy.select_action(_state(budget_progress=0.9, diversity=0.4), Regime.REFINE)
    repair_action = policy.select_action(_state(is_constrained=True, feasible_ratio=0.1), Regime.REPAIR)

    assert expand_action.operator_family is OperatorFamily.DE_LIKE
    assert expand_action.parametric_intent.prototype == "exploratory"
    assert refine_action.operator_family is OperatorFamily.SBX_LIKE
    assert refine_action.parametric_intent.prototype == "local_refine"
    assert repair_action.operator_family is OperatorFamily.SBX_LIKE
    assert repair_action.parametric_intent.prototype == "feasibility_biased"


def test_intent_prototypes_are_semantic_and_shared() -> None:
    names = available_intent_prototypes()
    exploratory = build_intent_prototype("exploratory")
    local_refine = build_intent_prototype("local_refine")

    assert names == ("exploratory", "balanced", "local_refine", "mutation_heavy", "feasibility_biased")
    assert exploratory.prototype == "exploratory"
    assert exploratory.exploration_strength > exploratory.locality
    assert local_refine.prototype == "local_refine"
    assert local_refine.locality > local_refine.exploration_strength


def test_adaptive_flat_operator_policy_tracks_rewarded_family() -> None:
    policy = AdaptiveFlatOperatorPolicy()
    state = _state(budget_progress=0.2, diversity=0.15, extent=0.15)

    action = policy.select_action(state, Regime.EXPAND)
    assert action.operator_family is OperatorFamily.DE_LIKE

    policy.update(state, action, outcome=None, credit=Credit(reward=1.0, bounded_reward=1.0))
    next_action = policy.select_action(state, Regime.EXPAND)

    assert next_action.operator_family is OperatorFamily.DE_LIKE


def test_adaptive_flat_parameter_policy_tracks_rewarded_prototype() -> None:
    policy = AdaptiveFlatParameterPolicy(fixed_family=OperatorFamily.SBX_LIKE)
    state = _state(budget_progress=0.2, diversity=0.12, extent=0.12, stagnation=0.2)

    action = policy.select_action(state, Regime.EXPAND)
    assert action.parametric_intent.prototype == "exploratory"

    policy.update(state, action, outcome=None, credit=Credit(reward=1.0, bounded_reward=1.0))
    next_action = policy.select_action(state, Regime.EXPAND)

    assert next_action.parametric_intent.prototype == "exploratory"


def test_adaptive_hierarchical_joint_policy_tracks_family_and_prototype() -> None:
    policy = AdaptiveHierarchicalJointPolicy()
    state = _state(budget_progress=0.2, diversity=0.12, extent=0.12, stagnation=0.2)

    action = policy.select_action(state, Regime.EXPAND)
    assert action.operator_family is OperatorFamily.DE_LIKE
    assert action.parametric_intent.prototype == "exploratory"

    policy.update(state, action, outcome=None, credit=Credit(reward=1.0, bounded_reward=1.0))
    next_action = policy.select_action(state, Regime.EXPAND)

    assert next_action.operator_family is OperatorFamily.DE_LIKE
    assert next_action.parametric_intent.prototype == "exploratory"


def test_adaptive_hierarchical_joint_policy_can_fix_family_without_disabling_prototypes() -> None:
    policy = AdaptiveHierarchicalJointPolicy(fixed_family=OperatorFamily.SBX_LIKE)
    state = _state(budget_progress=0.2, diversity=0.12, extent=0.12, stagnation=0.2)
    refine_state = _state(budget_progress=0.85, diversity=0.4, stagnation=0.1)

    action = policy.select_action(state, Regime.EXPAND)
    assert action.operator_family is OperatorFamily.SBX_LIKE
    assert action.parametric_intent.prototype == "exploratory"

    next_action = policy.select_action(refine_state, Regime.REFINE)

    assert next_action.operator_family is OperatorFamily.SBX_LIKE
    assert next_action.parametric_intent.prototype == "local_refine"
