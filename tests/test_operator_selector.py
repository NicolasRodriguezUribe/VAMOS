import numpy as np

from vamos.hyperheuristics.operator_selector import (
    EpsilonGreedyOperatorSelector,
    UCBOperatorSelector,
    compute_reward,
)


def test_epsilon_greedy_prefers_better_operator():
    rng = np.random.default_rng(0)
    selector = EpsilonGreedyOperatorSelector(2, epsilon=0.0, rng=rng)
    # operator 0 always gets reward 1, operator 1 gets 0
    for _ in range(50):
        idx = selector.select_operator()
        reward = 1.0 if idx == 0 else 0.0
        selector.update(idx, reward)
    counts = selector.counts
    assert counts[0] > counts[1]


def test_ucb_updates_values():
    selector = UCBOperatorSelector(2, c=1.0)
    selector.update(0, 1.0)
    selector.update(1, 0.0)
    choice = selector.select_operator()
    assert choice in (0, 1)
    assert selector.values[0] >= selector.values[1]


def test_compute_reward_modes():
    assert compute_reward(1.0, 2.0, "maximize") > 0
    assert compute_reward(1.0, 2.0, "minimize") < 0
