from __future__ import annotations

import pytest

from vamos.engine.algorithm.config import MOEADConfig
from vamos.engine.algorithm.moead import MOEAD
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _cfg(
    *,
    policy: str = "hierarchical_joint",
    fixed_family: str | None = None,
    credit_model: str = "simple_improvement",
) -> MOEADConfig:
    builder = (
        MOEADConfig.builder()
        .pop_size(6)
        .batch_size(1)
        .neighbor_size(3)
        .delta(0.9)
        .replace_limit(1)
        .crossover("sbx", prob=1.0, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .weight_vectors(divisions=6)
    )
    payload: dict[str, object] = {
        "enabled": True,
        "trace_level": "basic",
        "policy": policy,
        "credit_model": credit_model,
    }
    if fixed_family is not None:
        payload["fixed_family"] = fixed_family
    builder.online_control(**payload)
    return builder.build()


@pytest.mark.parametrize(
    ("policy", "fixed_family", "expected_cross_method", "expected_family"),
    [
        ("hierarchical_joint", None, "de", "de_like"),
        ("flat_parameter", "sbx_like", "sbx", "sbx_like"),
        ("adaptive_hierarchical_joint", None, "de", "de_like"),
        ("adaptive_flat_parameter", "sbx_like", "sbx", "sbx_like"),
    ],
)
def test_moead_online_control_runtime_wires_family_and_trace(
    policy: str,
    fixed_family: str | None,
    expected_cross_method: str,
    expected_family: str,
) -> None:
    problem = ZDT1Problem(n_var=4)
    algorithm = MOEAD(_cfg(policy=policy, fixed_family=fixed_family).to_dict(), kernel=NumPyKernel())
    _, eval_strategy, _, _ = algorithm._initialize_run(problem, ("max_evaluations", 20), seed=2, eval_strategy=None, live_viz=None)
    st = algorithm._st
    assert st is not None
    assert eval_strategy is not None

    X_off = algorithm.ask()
    assert st.current_cross_method == expected_cross_method
    eval_result = eval_strategy.evaluate(X_off, problem)
    algorithm.tell(eval_result, problem)

    trace = st.online_control_controller.trace_dicts() if st.online_control_controller is not None else []
    trace_rows = st.online_control_controller.trace_flat_dicts() if st.online_control_controller is not None else []
    run_summary = st.online_control_controller.run_summary() if st.online_control_controller is not None else {}
    assert len(trace) == 1
    assert len(trace_rows) == 1
    assert trace[0]["operator_family"] == expected_family
    assert trace[0]["intent_prototype"] is not None
    assert trace[0]["outcome"]["metadata"]["decoded_variation"]["cross_method"] == expected_cross_method
    assert trace_rows[0]["overhead_ms"] is not None
    assert 0.0 <= trace[0]["bounded_reward"] <= 1.0
    assert run_summary["steps"] == 1


def test_moead_online_control_cost_aware_summary_is_available() -> None:
    problem = ZDT1Problem(n_var=4)
    algorithm = MOEAD(_cfg(policy="adaptive_hierarchical_joint", credit_model="cost_aware").to_dict(), kernel=NumPyKernel())
    _, eval_strategy, _, _ = algorithm._initialize_run(problem, ("max_evaluations", 20), seed=2, eval_strategy=None, live_viz=None)
    st = algorithm._st
    assert st is not None
    assert eval_strategy is not None

    X_off = algorithm.ask()
    eval_result = eval_strategy.evaluate(X_off, problem)
    algorithm.tell(eval_result, problem)

    run_summary = st.online_control_controller.run_summary() if st.online_control_controller is not None else {}
    assert run_summary["credit_model"] == "cost_aware"
