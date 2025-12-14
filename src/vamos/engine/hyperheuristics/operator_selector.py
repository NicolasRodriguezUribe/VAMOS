from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np


class OperatorSelector(Protocol):
    def select_operator(self) -> int: ...

    def update(self, operator_index: int, reward: float, t: int | None = None) -> None: ...


@dataclass
class OperatorEntry:
    name: str
    weight: float = 1.0


class BanditOperatorSelector(OperatorSelector):
    """
    Base class for bandit-based operator selection.
    Tracks counts and value estimates; subclasses implement selection policy.
    """

    def __init__(self, n_ops: int):
        if n_ops <= 0:
            raise ValueError("n_ops must be positive.")
        self.counts = np.zeros(n_ops, dtype=int)
        self.values = np.zeros(n_ops, dtype=float)
        self.t = 0

    def update(self, operator_index: int, reward: float, t: int | None = None) -> None:
        self.t = self.t + 1 if t is None else t
        self.counts[operator_index] += 1
        n = self.counts[operator_index]
        q = self.values[operator_index]
        self.values[operator_index] = q + (reward - q) / max(1, n)


class EpsilonGreedyOperatorSelector(BanditOperatorSelector):
    def __init__(self, n_ops: int, epsilon: float = 0.1, rng: np.random.Generator | None = None):
        super().__init__(n_ops)
        self.epsilon = float(epsilon)
        self.rng = rng or np.random.default_rng()

    def select_operator(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, len(self.counts)))
        return int(np.argmax(self.values))


class UCBOperatorSelector(BanditOperatorSelector):
    def __init__(self, n_ops: int, c: float = 1.0):
        super().__init__(n_ops)
        self.c = float(c)

    def select_operator(self) -> int:
        self.t += 1
        total = max(1, self.t)
        bonus = self.c * np.sqrt(np.log(total) / (self.counts + 1e-9))
        ucb = self.values + bonus
        return int(np.argmax(ucb))


def make_operator_selector(method: str, n_ops: int, **kwargs) -> OperatorSelector:
    method = method.lower()
    if method in {"epsilon_greedy", "egreedy", "eps"}:
        return EpsilonGreedyOperatorSelector(n_ops, epsilon=kwargs.get("epsilon", 0.1), rng=kwargs.get("rng"))
    if method in {"ucb", "ucb1"}:
        return UCBOperatorSelector(n_ops, c=kwargs.get("c", 1.0))
    raise ValueError(f"Unknown operator selector method '{method}'.")


def compute_reward(old_value: float, new_value: float, mode: str = "maximize") -> float:
    """
    Compute reward based on improvement in indicator value.
    """
    if mode == "maximize":
        return new_value - old_value
    return old_value - new_value

