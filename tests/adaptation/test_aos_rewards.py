from __future__ import annotations

from vamos.adaptation.aos.rewards import (
    aggregate_reward,
    nd_insertion_rate,
    survival_rate,
)


def test_survival_rate_edge_cases() -> None:
    assert survival_rate(0, 0) == 0.0
    assert survival_rate(5, 0) == 0.0
    assert survival_rate(5, 10) == 0.5
    assert survival_rate(12, 10) == 1.0
    assert survival_rate(-3, 10) == 0.0


def test_nd_insertion_rate_edge_cases() -> None:
    assert nd_insertion_rate(0, 0) == 0.0
    assert nd_insertion_rate(4, 0) == 0.0
    assert nd_insertion_rate(3, 6) == 0.5
    assert nd_insertion_rate(8, 6) == 1.0
    assert nd_insertion_rate(-2, 6) == 0.0


def test_aggregate_reward_components() -> None:
    summary = aggregate_reward(
        "combined",
        survival_rate=0.25,
        nd_rate=0.75,
        hv_delta_rate=0.5,
        weights={"survival": 0.5, "nd_insertions": 0.5, "hv_delta": 0.0},
    )
    assert 0.0 <= summary.reward <= 1.0
    assert summary.reward_survival == 0.25
    assert summary.reward_nd_insertions == 0.75
    assert summary.reward_hv_delta == 0.5

    scoped = aggregate_reward("survival", 0.2, 0.9, hv_delta_rate=0.0)
    assert scoped.reward == 0.2
