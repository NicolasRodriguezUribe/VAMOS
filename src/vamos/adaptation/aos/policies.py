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


__all__ = [
    "OperatorBanditPolicy",
    "UCBPolicy",
    "EpsGreedyPolicy",
    "EXP3Policy",
]
