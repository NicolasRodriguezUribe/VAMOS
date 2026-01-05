"""
Bandit policies for adaptive operator selection.
"""

from __future__ import annotations

import math
import random
from typing import Protocol


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _argmax(values: list[float]) -> int:
    best_idx = 0
    best_val = values[0]
    for idx, val in enumerate(values[1:], start=1):
        if val > best_val:
            best_idx = idx
            best_val = val
    return best_idx


def _select_min_usage(counts: list[int], min_usage: int) -> int | None:
    if min_usage <= 0:
        return None
    min_count = min(counts)
    if min_count >= min_usage:
        return None
    for idx, count in enumerate(counts):
        if count == min_count:
            return idx
    return None


class OperatorBanditPolicy(Protocol):
    """
    Protocol for AOS bandit policies.
    """

    def select_arm(self) -> int: ...

    def update(self, arm_index: int, reward: float) -> None: ...

    def probs(self) -> list[float]: ...

    def counts(self) -> list[int]: ...


class UCBPolicy:
    """
    Upper Confidence Bound (UCB1) policy with optional warm-up usage.
    """

    def __init__(self, n_arms: int, *, c: float = 1.0, min_usage: int = 1):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive.")
        self.n_arms = int(n_arms)
        self.c = float(c)
        self.min_usage = int(min_usage)
        self._counts = [0] * self.n_arms
        self._values = [0.0] * self.n_arms

    def select_arm(self) -> int:
        warm = _select_min_usage(self._counts, self.min_usage)
        if warm is not None:
            return warm
        total = sum(self._counts)
        if total <= 0:
            return 0
        log_total = math.log(max(1, total))
        best_idx = 0
        best_score = float("-inf")
        for idx in range(self.n_arms):
            count = self._counts[idx]
            if count <= 0:
                return idx
            bonus = self.c * math.sqrt(log_total / count)
            score = self._values[idx] + bonus
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def update(self, arm_index: int, reward: float) -> None:
        reward = _clamp01(reward)
        self._counts[arm_index] += 1
        count = self._counts[arm_index]
        prev = self._values[arm_index]
        self._values[arm_index] = prev + (reward - prev) / float(count)

    def probs(self) -> list[float]:
        total = sum(self._counts)
        if total <= 0:
            return [1.0 / self.n_arms] * self.n_arms
        log_total = math.log(max(1, total))
        scores = []
        for idx in range(self.n_arms):
            count = max(1, self._counts[idx])
            bonus = self.c * math.sqrt(log_total / count)
            scores.append(self._values[idx] + bonus)
        score_sum = sum(scores)
        if score_sum <= 0.0:
            return [1.0 / self.n_arms] * self.n_arms
        return [score / score_sum for score in scores]

    def counts(self) -> list[int]:
        return list(self._counts)


class EpsGreedyPolicy:
    """
    Epsilon-greedy policy with deterministic RNG seed.
    """

    def __init__(
        self,
        n_arms: int,
        *,
        epsilon: float = 0.1,
        rng_seed: int | None = None,
        min_usage: int = 1,
    ):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive.")
        self.n_arms = int(n_arms)
        self.epsilon = float(epsilon)
        self.min_usage = int(min_usage)
        self._rng = random.Random(rng_seed)
        self._counts = [0] * self.n_arms
        self._values = [0.0] * self.n_arms

    def select_arm(self) -> int:
        warm = _select_min_usage(self._counts, self.min_usage)
        if warm is not None:
            return warm
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(self.n_arms)
        return _argmax(self._values)

    def update(self, arm_index: int, reward: float) -> None:
        reward = _clamp01(reward)
        self._counts[arm_index] += 1
        count = self._counts[arm_index]
        prev = self._values[arm_index]
        self._values[arm_index] = prev + (reward - prev) / float(count)

    def probs(self) -> list[float]:
        base = self.epsilon / float(self.n_arms)
        probs = [base] * self.n_arms
        greedy = _argmax(self._values)
        probs[greedy] += 1.0 - self.epsilon
        return probs

    def counts(self) -> list[int]:
        return list(self._counts)


class EXP3Policy:
    """
    EXP3 policy with gamma exploration and deterministic RNG seed.
    """

    def __init__(self, n_arms: int, *, gamma: float = 0.2, rng_seed: int | None = None):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive.")
        if gamma <= 0.0 or gamma > 1.0:
            raise ValueError("gamma must be in (0, 1].")
        self.n_arms = int(n_arms)
        self.gamma = float(gamma)
        self._rng = random.Random(rng_seed)
        self._weights = [1.0] * self.n_arms
        self._counts = [0] * self.n_arms

    def probs(self) -> list[float]:
        total = sum(self._weights)
        if total <= 0.0:
            return [1.0 / self.n_arms] * self.n_arms
        base = [w / total for w in self._weights]
        probs = [(1.0 - self.gamma) * b + self.gamma / self.n_arms for b in base]
        total_p = sum(probs)
        if total_p <= 0.0:
            return [1.0 / self.n_arms] * self.n_arms
        return [p / total_p for p in probs]

    def select_arm(self) -> int:
        probs = self.probs()
        draw = self._rng.random()
        accum = 0.0
        for idx, prob in enumerate(probs):
            accum += prob
            if draw <= accum:
                return idx
        return self.n_arms - 1

    def update(self, arm_index: int, reward: float) -> None:
        reward = _clamp01(reward)
        probs = self.probs()
        prob = probs[arm_index]
        if prob <= 0.0:
            return
        x_hat = reward / prob
        factor = math.exp((self.gamma * x_hat) / self.n_arms)
        self._weights[arm_index] *= factor
        self._counts[arm_index] += 1

    def counts(self) -> list[int]:
        return list(self._counts)


class ThompsonSamplingPolicy:
    """
    Thompson Sampling policy with Beta priors for [0,1] bounded rewards.

    Uses Beta(alpha, beta) distribution where:
    - alpha = 1 + sum of rewards (successes)
    - beta = 1 + sum of (1 - reward) (failures)

    Supports optional sliding window for non-stationary environments.
    """

    def __init__(
        self,
        n_arms: int,
        *,
        rng_seed: int | None = None,
        min_usage: int = 1,
        window_size: int = 0,
    ):
        """
        Args:
            n_arms: Number of arms.
            rng_seed: Random seed for reproducibility.
            min_usage: Minimum pulls per arm before Thompson selection.
            window_size: If > 0, only use last `window_size` rewards per arm.
        """
        if n_arms <= 0:
            raise ValueError("n_arms must be positive.")
        self.n_arms = int(n_arms)
        self.min_usage = int(min_usage)
        self.window_size = int(window_size)
        self._rng = random.Random(rng_seed)
        self._counts = [0] * self.n_arms
        # Store reward history for sliding window
        self._reward_history: list[list[float]] = [[] for _ in range(self.n_arms)]

    def _get_alpha_beta(self, arm_index: int) -> tuple[float, float]:
        """Compute Beta parameters from reward history."""
        rewards = self._reward_history[arm_index]
        if self.window_size > 0 and len(rewards) > self.window_size:
            rewards = rewards[-self.window_size:]
        if not rewards:
            return 1.0, 1.0
        alpha = 1.0 + sum(rewards)
        beta = 1.0 + sum(1.0 - r for r in rewards)
        return alpha, beta

    def select_arm(self) -> int:
        # Warm-up phase
        warm = _select_min_usage(self._counts, self.min_usage)
        if warm is not None:
            return warm

        # Sample from Beta distribution for each arm
        samples = []
        for idx in range(self.n_arms):
            alpha, beta = self._get_alpha_beta(idx)
            # Use inverse transform sampling approximation for Beta
            sample = self._sample_beta(alpha, beta)
            samples.append(sample)

        return _argmax(samples)

    def _sample_beta(self, alpha: float, beta: float) -> float:
        """Sample from Beta(alpha, beta) using random.betavariate."""
        try:
            return self._rng.betavariate(alpha, beta)
        except ValueError:
            # Fallback for edge cases
            return 0.5

    def update(self, arm_index: int, reward: float) -> None:
        reward = _clamp01(reward)
        self._counts[arm_index] += 1
        self._reward_history[arm_index].append(reward)

    def probs(self) -> list[float]:
        """Return mean of Beta distribution as probability proxy."""
        means = []
        for idx in range(self.n_arms):
            alpha, beta = self._get_alpha_beta(idx)
            mean = alpha / (alpha + beta)
            means.append(mean)
        total = sum(means)
        if total <= 0.0:
            return [1.0 / self.n_arms] * self.n_arms
        return [m / total for m in means]

    def counts(self) -> list[int]:
        return list(self._counts)


class SlidingWindowUCBPolicy:
    """
    UCB1 policy with sliding window for non-stationary environments.

    Only considers the last `window_size` rewards per arm when computing
    the UCB score.
    """

    def __init__(
        self,
        n_arms: int,
        *,
        c: float = 1.0,
        min_usage: int = 1,
        window_size: int = 50,
    ):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive.")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        self.n_arms = int(n_arms)
        self.c = float(c)
        self.min_usage = int(min_usage)
        self.window_size = int(window_size)
        self._counts = [0] * self.n_arms
        self._reward_history: list[list[float]] = [[] for _ in range(self.n_arms)]

    def _windowed_mean(self, arm_index: int) -> float:
        rewards = self._reward_history[arm_index]
        if not rewards:
            return 0.0
        window = rewards[-self.window_size:] if len(rewards) > self.window_size else rewards
        return sum(window) / len(window)

    def _windowed_count(self, arm_index: int) -> int:
        rewards = self._reward_history[arm_index]
        return min(len(rewards), self.window_size)

    def select_arm(self) -> int:
        warm = _select_min_usage(self._counts, self.min_usage)
        if warm is not None:
            return warm

        total = sum(self._windowed_count(i) for i in range(self.n_arms))
        if total <= 0:
            return 0

        log_total = math.log(max(1, total))
        best_idx = 0
        best_score = float("-inf")

        for idx in range(self.n_arms):
            w_count = self._windowed_count(idx)
            if w_count <= 0:
                return idx
            bonus = self.c * math.sqrt(log_total / w_count)
            score = self._windowed_mean(idx) + bonus
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def update(self, arm_index: int, reward: float) -> None:
        reward = _clamp01(reward)
        self._counts[arm_index] += 1
        self._reward_history[arm_index].append(reward)

    def probs(self) -> list[float]:
        total = sum(self._windowed_count(i) for i in range(self.n_arms))
        if total <= 0:
            return [1.0 / self.n_arms] * self.n_arms

        log_total = math.log(max(1, total))
        scores = []
        for idx in range(self.n_arms):
            w_count = max(1, self._windowed_count(idx))
            bonus = self.c * math.sqrt(log_total / w_count)
            scores.append(self._windowed_mean(idx) + bonus)

        score_sum = sum(scores)
        if score_sum <= 0.0:
            return [1.0 / self.n_arms] * self.n_arms
        return [score / score_sum for score in scores]

    def counts(self) -> list[int]:
        return list(self._counts)


__all__ = [
    "OperatorBanditPolicy",
    "UCBPolicy",
    "EpsGreedyPolicy",
    "EXP3Policy",
    "ThompsonSamplingPolicy",
    "SlidingWindowUCBPolicy",
]
