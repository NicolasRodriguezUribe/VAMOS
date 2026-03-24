from __future__ import annotations

import pytest

from vamos import optimize
from vamos.engine.adaptation.online_control import HierarchicalAction, OperatorFamily, ParametricIntent, Regime
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _cfg(
    *,
    enabled: bool,
    policy: str = "hierarchical_joint",
    router: str = "heuristic",
    fixed_family: str | None = None,
    credit_model: str = "simple_improvement",
) -> NSGAIIConfig:
    builder = (
        NSGAIIConfig.builder()
        .pop_size(6)
        .offspring_size(6)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("polynomial", prob=0.1, eta=20.0)
        .selection("tournament", size=2)
        .result_mode("population")
    )
    if enabled:
        payload: dict[str, object] = {
            "enabled": True,
            "trace_level": "basic",
            "router": router,
            "policy": policy,
            "credit_model": credit_model,
        }
        if fixed_family is not None:
            payload["fixed_family"] = fixed_family
        builder.online_control(**payload)
    return builder.build()


def _initialized_algo(config: NSGAIIConfig) -> tuple[NSGAII, ZDT1Problem]:
    problem = ZDT1Problem(n_var=4)
    algorithm = NSGAII(config.to_dict(), kernel=NumPyKernel())
    algorithm._initialize_run(problem, ("max_evaluations", 18), seed=1, eval_strategy=None, live_viz=None)
    return algorithm, problem


def test_nsgaii_online_control_disabled_keeps_payload_clean() -> None:
    problem = ZDT1Problem(n_var=4)
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=_cfg(enabled=False),
        termination=("max_evaluations", 12),
        seed=1,
        engine="numpy",
    )
    assert "online_control" not in result.data


def test_nsgaii_decode_action_translates_intent_into_real_parameters() -> None:
    algorithm, _ = _initialized_algo(_cfg(enabled=True))
    st = algorithm._st
    assert st is not None
    assert st.online_control_adapter is not None

    local_action = HierarchicalAction(
        regime=Regime.REFINE,
        operator_family=OperatorFamily.SBX_LIKE,
        parametric_intent=ParametricIntent(
            exploration_strength=0.1,
            locality=0.9,
            mutation_strength=0.2,
            feasibility_bias=0.7,
        ),
    )
    broad_action = HierarchicalAction(
        regime=Regime.EXPAND,
        operator_family=OperatorFamily.SBX_LIKE,
        parametric_intent=ParametricIntent(
            exploration_strength=0.9,
            locality=0.2,
            mutation_strength=0.8,
            feasibility_bias=0.2,
        ),
    )
    de_action = HierarchicalAction(
        regime=Regime.EXPAND,
        operator_family=OperatorFamily.DE_LIKE,
        parametric_intent=ParametricIntent(
            exploration_strength=0.9,
            locality=0.1,
            mutation_strength=0.7,
            feasibility_bias=0.1,
        ),
    )

    local_pipeline = st.online_control_adapter.decode_action(local_action, st)
    broad_pipeline = st.online_control_adapter.decode_action(broad_action, st)
    de_pipeline = st.online_control_adapter.decode_action(de_action, st)

    assert local_pipeline.cross_method == "sbx"
    assert broad_pipeline.cross_method == "sbx"
    assert de_pipeline.cross_method == "de"
    assert float(local_pipeline.cross_params["eta"]) > float(broad_pipeline.cross_params["eta"])
    assert float(local_pipeline.mut_params["eta"]) > float(broad_pipeline.mut_params["eta"])
    assert float(local_pipeline.mut_params["prob"]) < float(broad_pipeline.mut_params["prob"])


@pytest.mark.parametrize(
    ("policy", "router", "fixed_family", "expected_first_family", "expected_first_regime"),
    [
        ("flat_operator", "heuristic", None, "de_like", "expand"),
        ("flat_parameter", "heuristic", "sbx_like", "sbx_like", "expand"),
        ("hierarchical_joint", "heuristic", None, "de_like", "expand"),
        ("adaptive_flat_operator", "heuristic", None, "de_like", "expand"),
        ("adaptive_flat_parameter", "heuristic", "sbx_like", "sbx_like", "expand"),
        ("adaptive_hierarchical_joint", "heuristic", None, "de_like", "expand"),
        ("adaptive_hierarchical_joint", "static_refine", None, "sbx_like", "refine"),
        ("adaptive_hierarchical_joint", "heuristic", "sbx_like", "sbx_like", "expand"),
    ],
)
def test_nsgaii_online_control_records_semantic_trace(
    policy: str,
    router: str,
    fixed_family: str | None,
    expected_first_family: str,
    expected_first_regime: str,
) -> None:
    problem = ZDT1Problem(n_var=4)
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=_cfg(enabled=True, policy=policy, router=router, fixed_family=fixed_family),
        termination=("max_evaluations", 18),
        seed=1,
        engine="numpy",
    )

    payload = result.data["online_control"]
    trace = payload["trace"]
    trace_rows = payload["trace_rows"]
    run_summary = payload["run_summary"]
    assert payload["enabled"] is True
    assert len(trace) == 2
    assert len(trace_rows) == 2
    first = trace[0]
    intent = first["parametric_intent"]

    assert first["search_state"]["host"] == "nsgaii"
    assert first["regime"] == expected_first_regime
    assert first["operator_family"] == expected_first_family
    assert first["intent_prototype"] is not None
    assert "decoded_variation" in first["outcome"]["metadata"]
    assert 0.0 <= first["bounded_reward"] <= 1.0
    assert trace_rows[0]["overhead_ms"] is not None
    assert payload["summary"]
    assert run_summary["steps"] == 2
    assert "family_shares" in run_summary
    assert "family_switches" in run_summary
    assert run_summary["runtime_profile"]["decode_time_ms"] >= 0.0
    assert run_summary["policy_select_time_ms"] >= 0.0
    if policy == "flat_operator":
        assert intent["exploration_strength"] == 0.5
        assert intent["locality"] == 0.5
        assert intent["mutation_strength"] == 0.5
    else:
        assert intent["prototype"] is not None


def test_nsgaii_cost_aware_credit_changes_run_summary() -> None:
    problem = ZDT1Problem(n_var=4)
    result = optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=_cfg(enabled=True, policy="adaptive_hierarchical_joint", credit_model="cost_aware"),
        termination=("max_evaluations", 18),
        seed=1,
        engine="numpy",
    )

    online = result.data["online_control"]
    assert online["run_summary"]["credit_model"] == "cost_aware"
    assert online["trace_rows"][0]["credit_model"] == "cost_aware"
