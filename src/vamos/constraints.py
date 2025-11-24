from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ConstraintInfo:
    G: np.ndarray
    cv: np.ndarray
    feasible_mask: np.ndarray


def compute_constraint_info(G: np.ndarray, eps: float = 0.0) -> ConstraintInfo:
    """
    Compute aggregate constraint violation and feasibility mask.

    Args:
        G: (n_points, n_constr) constraint values, <=0 means satisfied.
        eps: tolerance. Constraints <= eps are treated as satisfied.
    """
    if G is None:
        raise ValueError("G must be provided; use an array of zeros when unconstrained.")
    G = np.asarray(G, dtype=float)
    if G.ndim != 2:
        raise ValueError("G must be a 2D array of shape (n_points, n_constr).")
    positive = np.maximum(G - eps, 0.0)
    cv = np.sum(positive, axis=1)
    feasible = np.all(G <= eps, axis=1)
    return ConstraintInfo(G=G, cv=cv, feasible_mask=feasible)


class ConstraintHandlingStrategy(ABC):
    """
    Base class for constraint handling in VAMOS.
    """

    @abstractmethod
    def rank(self, F: np.ndarray, G: np.ndarray | None) -> np.ndarray:
        """
        Return a scalar key per solution; lower is better.
        """


def _aggregate_objectives(F: np.ndarray, mode: str) -> np.ndarray:
    if mode == "sum":
        return np.sum(F, axis=1)
    if mode == "max":
        return np.max(F, axis=1)
    if mode == "none":
        return np.zeros(F.shape[0], dtype=float)
    raise ValueError(f"Unknown objective aggregator '{mode}'.")


class FeasibilityFirstStrategy(ConstraintHandlingStrategy):
    def __init__(self, objective_aggregator: str = "sum"):
        self.objective_aggregator = objective_aggregator

    def rank(self, F: np.ndarray, G: np.ndarray | None) -> np.ndarray:
        F = np.asarray(F, dtype=float)
        if G is None:
            return _aggregate_objectives(F, self.objective_aggregator)
        info = compute_constraint_info(G)
        agg = _aggregate_objectives(F, self.objective_aggregator)
        infeasible_penalty = info.cv
        # Feasible get 0 prefix, infeasible 1 + cv to stay worse
        return np.where(info.feasible_mask, agg, agg.max(initial=0.0) + 1.0 + infeasible_penalty)


class PenaltyCVStrategy(ConstraintHandlingStrategy):
    def __init__(self, penalty_lambda: float = 1000.0, objective_aggregator: str = "sum"):
        if penalty_lambda <= 0:
            raise ValueError("penalty_lambda must be positive.")
        self.penalty_lambda = float(penalty_lambda)
        self.objective_aggregator = objective_aggregator

    def rank(self, F: np.ndarray, G: np.ndarray | None) -> np.ndarray:
        F = np.asarray(F, dtype=float)
        agg = _aggregate_objectives(F, self.objective_aggregator)
        if G is None:
            return agg
        info = compute_constraint_info(G)
        return agg + self.penalty_lambda * info.cv


class CVAsObjectiveStrategy(ConstraintHandlingStrategy):
    def __init__(self, objective_aggregator: str = "sum", eps: float = 1e-6):
        self.objective_aggregator = objective_aggregator
        self.eps = float(eps)

    def rank(self, F: np.ndarray, G: np.ndarray | None) -> np.ndarray:
        F = np.asarray(F, dtype=float)
        agg = _aggregate_objectives(F, self.objective_aggregator)
        if G is None:
            return agg
        info = compute_constraint_info(G)
        return info.cv + self.eps * agg


class EpsilonConstraintStrategy(ConstraintHandlingStrategy):
    def __init__(self, epsilon: float = 0.0, objective_aggregator: str = "sum"):
        self.epsilon = float(epsilon)
        self.objective_aggregator = objective_aggregator

    def rank(self, F: np.ndarray, G: np.ndarray | None) -> np.ndarray:
        F = np.asarray(F, dtype=float)
        if G is None:
            return _aggregate_objectives(F, self.objective_aggregator)
        info = compute_constraint_info(G, eps=self.epsilon)
        agg = _aggregate_objectives(F, self.objective_aggregator)
        infeasible_penalty = info.cv
        return np.where(info.feasible_mask, agg, agg.max(initial=0.0) + 1.0 + infeasible_penalty)


def get_constraint_strategy(name: str, **kwargs) -> ConstraintHandlingStrategy:
    key = name.lower()
    if key == "feasibility_first":
        return FeasibilityFirstStrategy(**kwargs)
    if key == "penalty_cv":
        return PenaltyCVStrategy(**kwargs)
    if key == "cv_as_objective":
        return CVAsObjectiveStrategy(**kwargs)
    if key == "epsilon":
        return EpsilonConstraintStrategy(**kwargs)
    raise ValueError(f"Unknown constraint strategy '{name}'.")
