from __future__ import annotations

from vamos.adaptation.aos.policies import EpsGreedyPolicy, EXP3Policy, UCBPolicy


def test_ucb_policy_min_usage() -> None:
    policy = UCBPolicy(n_arms=3, c=1.0, min_usage=2)
    pulls = []
    for _ in range(6):
        idx = policy.select_arm()
        pulls.append(idx)
        policy.update(idx, 0.0)
    assert pulls == [0, 1, 2, 0, 1, 2]


def test_eps_greedy_deterministic_with_seed() -> None:
    def run_sequence(seed: int) -> list[int]:
        policy = EpsGreedyPolicy(n_arms=3, epsilon=0.5, rng_seed=seed, min_usage=0)
        seq: list[int] = []
        for _ in range(20):
            idx = policy.select_arm()
            seq.append(idx)
            reward = 1.0 if idx == 0 else 0.0
            policy.update(idx, reward)
        return seq

    assert run_sequence(123) == run_sequence(123)


def test_exp3_probs_and_response() -> None:
    policy = EXP3Policy(n_arms=2, gamma=0.2, rng_seed=7)
    initial = policy.probs()
    assert abs(sum(initial) - 1.0) < 1e-9
    assert all(p >= 0.0 for p in initial)

    for _ in range(10):
        policy.update(0, 1.0)
    updated = policy.probs()
    assert abs(sum(updated) - 1.0) < 1e-9
    assert all(p >= 0.0 for p in updated)
    assert updated[0] > initial[0]
