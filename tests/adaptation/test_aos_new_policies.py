"""Tests for new AOS policies: Thompson Sampling and Sliding Window UCB."""

from __future__ import annotations

from vamos.adaptation.aos.policies import ThompsonSamplingPolicy, SlidingWindowUCBPolicy


def test_thompson_sampling_basic():
    """Test ThompsonSamplingPolicy basic functionality."""
    policy = ThompsonSamplingPolicy(n_arms=3, rng_seed=42, min_usage=1)

    # Warm-up phase
    pulls = []
    for _ in range(3):
        idx = policy.select_arm()
        pulls.append(idx)
        policy.update(idx, 0.5)

    # Should have tried each arm at least once
    assert set(pulls) == {0, 1, 2}


def test_thompson_sampling_learns():
    """Test that Thompson Sampling learns to prefer higher reward arms."""
    policy = ThompsonSamplingPolicy(n_arms=2, rng_seed=123, min_usage=1)

    # Warm up
    policy.update(0, 0.0)
    policy.update(1, 0.0)

    # Train: arm 0 always rewards 0.9, arm 1 always rewards 0.1
    for _ in range(50):
        idx = policy.select_arm()
        reward = 0.9 if idx == 0 else 0.1
        policy.update(idx, reward)

    # After training, arm 0 should be selected more often
    counts = policy.counts()
    assert counts[0] > counts[1], f"Expected arm 0 to be preferred, got {counts}"


def test_thompson_sampling_with_window():
    """Test Thompson Sampling with sliding window."""
    policy = ThompsonSamplingPolicy(n_arms=2, rng_seed=7, min_usage=1, window_size=5)

    # Initial: arm 0 is best
    for _ in range(10):
        policy.update(0, 0.9)
        policy.update(1, 0.1)

    # Shift: now arm 1 is best
    for _ in range(10):
        policy.update(0, 0.1)
        policy.update(1, 0.9)

    # With window_size=5, policy should adapt to new best
    probs = policy.probs()
    assert probs[1] > probs[0], f"With sliding window, arm 1 should be preferred, got {probs}"


def test_thompson_sampling_deterministic_seed():
    """Test that Thompson Sampling is deterministic with same seed."""

    def run(seed):
        policy = ThompsonSamplingPolicy(n_arms=3, rng_seed=seed, min_usage=0)
        selections = []
        for _ in range(10):
            idx = policy.select_arm()
            selections.append(idx)
            policy.update(idx, 0.5)
        return selections

    assert run(42) == run(42)


def test_sliding_window_ucb_basic():
    """Test SlidingWindowUCBPolicy basic functionality."""
    policy = SlidingWindowUCBPolicy(n_arms=3, c=1.0, min_usage=1, window_size=10)

    # Warm-up phase
    for _ in range(3):
        idx = policy.select_arm()
        policy.update(idx, 0.5)

    # Should have tried each arm
    counts = policy.counts()
    assert sum(counts) == 3


def test_sliding_window_ucb_adapts():
    """Test that sliding window UCB adapts to changing rewards."""
    policy = SlidingWindowUCBPolicy(n_arms=2, c=0.5, min_usage=1, window_size=5)

    # Phase 1: arm 0 is best
    for _ in range(10):
        policy.update(0, 1.0)
        policy.update(1, 0.0)

    # Phase 2: arm 1 is now best
    for _ in range(10):
        policy.update(0, 0.0)
        policy.update(1, 1.0)

    # With window_size=5, windowed mean for arm 1 should be higher
    probs = policy.probs()
    assert probs[1] > probs[0], f"With sliding window, arm 1 should be preferred, got {probs}"


def test_sliding_window_ucb_probs_sum_to_one():
    """Test that probs() returns valid probability distribution."""
    policy = SlidingWindowUCBPolicy(n_arms=3, window_size=10)

    for _ in range(5):
        idx = policy.select_arm()
        policy.update(idx, 0.5)

    probs = policy.probs()
    assert abs(sum(probs) - 1.0) < 1e-9
    assert all(p >= 0.0 for p in probs)
