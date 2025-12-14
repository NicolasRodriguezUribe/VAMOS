from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol

import numpy as np

from vamos.engine.hyperheuristics.operator_selector import make_operator_selector, compute_reward
from vamos.engine.hyperheuristics.indicator import IndicatorEvaluator


class PortfolioAlgorithm(Protocol):
    def step(self, evaluations: int) -> None: ...

    def current_front(self) -> np.ndarray: ...


@dataclass
class PortfolioEntry:
    name: str
    algo: PortfolioAlgorithm
    n_selected: int = 0
    value_estimate: float = 0.0


class AlgorithmPortfolio:
    """
    Minimal bandit-based portfolio over algorithm instances that expose step()/current_front().
    """

    def __init__(
        self,
        entries: List[PortfolioEntry],
        indicator: str = "hv",
        mode: str = "maximize",
        selector_method: str = "epsilon_greedy",
        selector_kwargs: dict | None = None,
    ):
        if not entries:
            raise ValueError("Portfolio requires at least one algorithm entry.")
        self.entries = entries
        self.selector = make_operator_selector(selector_method, len(entries), **(selector_kwargs or {}))
        self.indicator_eval = IndicatorEvaluator(indicator, reference_point=None, mode=mode)

    def combined_front(self) -> np.ndarray:
        fronts = [e.algo.current_front() for e in self.entries if e.algo.current_front() is not None]
        if not fronts:
            return np.empty((0, 0))
        union = np.vstack(fronts)
        # Non-dominated filtering: simple O(n^2) fallback
        mask = np.ones(union.shape[0], dtype=bool)
        for i in range(union.shape[0]):
            if not mask[i]:
                continue
            for j in range(union.shape[0]):
                if i == j or not mask[j]:
                    continue
                if np.all(union[j] <= union[i]) and np.any(union[j] < union[i]):
                    mask[i] = False
                    break
        return union[mask]

    def step(self, step_evaluations: int) -> None:
        idx = self.selector.select_operator()
        entry = self.entries[idx]
        before = self.indicator_eval.compute(self.combined_front()) if self.combined_front().size else 0.0
        entry.algo.step(step_evaluations)
        after = self.indicator_eval.compute(self.combined_front()) if self.combined_front().size else before
        reward = compute_reward(before, after, self.indicator_eval.mode)
        self.selector.update(idx, reward)
