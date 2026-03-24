from __future__ import annotations

from vamos.engine.adaptation.online_control import (
    OnlineControlController,
    OperatorFamily,
    Outcome,
    Regime,
    SearchState,
    build_online_control_controller,
)


def _search_state(step_index: int = 0) -> SearchState:
    return SearchState(
        host="nsgaii",
        step_index=step_index,
        generation=step_index,
        evaluations=6 + step_index * 6,
        population_size=6,
        objective_count=2,
        decision_dim=4,
        evaluation_budget=30,
        budget_progress=0.2,
        extent=0.18,
        diversity=0.12,
        stagnation=0.0,
        quality_indicator=-0.4,
        available_families=(OperatorFamily.SBX_LIKE, OperatorFamily.DE_LIKE),
    )


def test_controller_lifecycle_records_semantic_trace_rows() -> None:
    controller = OnlineControlController()
    controller.start_step(_search_state(step_index=2))
    action = controller.select_action()
    credit = controller.finalize_step(
        Outcome(
            success=True,
            reward_hint=0.5,
            survivor_ratio=0.5,
            accepted_ratio=0.5,
            nd_insertions_ratio=0.2,
            extent=0.24,
            extent_gain=0.06,
            quality_indicator=-0.3,
            quality_delta=0.1,
            overhead_ms=0.7,
            metadata={"survivors": 3},
        )
    )
    trace = controller.trace_dicts()
    payload = controller.result_payload()

    assert action.regime is Regime.EXPAND
    assert action.operator_family is OperatorFamily.DE_LIKE
    assert credit.metadata["model"] == "simple_improvement"
    assert 0.0 < credit.reward < 0.5
    assert len(trace) == 1
    assert trace[0]["step_index"] == 2
    assert trace[0]["generation"] == 2
    assert trace[0]["regime"] == "expand"
    assert trace[0]["operator_family"] == "de_like"
    assert trace[0]["intent_prototype"] in {"balanced", "exploratory", "mutation_heavy"}
    assert trace[0]["outcome"]["metadata"]["survivors"] == 3
    assert 0.0 < trace[0]["reward"] < 0.5
    assert payload["trace_rows"][0]["overhead_ms"] == 0.7
    assert payload["summary"]
    assert payload["run_summary"]["steps"] == 1


def test_build_online_control_controller_supports_static_router_bypass() -> None:
    controller = build_online_control_controller(
        {
            "enabled": True,
            "router": "static_refine",
            "policy": "adaptive_hierarchical_joint",
            "credit_model": "simple_improvement",
        }
    )

    assert controller is not None
    controller.start_step(_search_state(step_index=1))
    action = controller.select_action()

    assert action.regime is Regime.REFINE
