"""
Algorithm configuration space with conditional parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .parameters import BaseParam, ConditionalBlock


@dataclass
class AlgorithmConfigSpace:
    """
    Declarative hyperparameter space for an algorithm.
    """

    algorithm_name: str
    params: List[BaseParam]
    conditionals: List[ConditionalBlock] | None = None

    def _active_conditionals(self, assignment: Dict[str, Any]) -> List[ConditionalBlock]:
        active: List[ConditionalBlock] = []
        for block in self.conditionals or []:
            if assignment.get(block.parent_name) == block.parent_value:
                active.append(block)
        return active

    def flatten(self, assignment: Dict[str, Any] | None = None) -> List[BaseParam]:
        """
        Return the list of active parameters given a partial assignment.
        If assignment is None, only top-level params are returned.
        """
        if assignment is None:
            return list(self.params)
        active = list(self.params)
        for block in self._active_conditionals(assignment):
            active.extend(block.params)
        return active

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Sample a concrete assignment dict for all active params.
        """
        assignment: Dict[str, Any] = {}
        for p in self.params:
            assignment[p.name] = p.sample(rng)
        for block in self._active_conditionals(assignment):
            for p in block.params:
                assignment[p.name] = p.sample(rng)
        return assignment

    def to_unit_vector(self, assignment: Dict[str, Any]) -> np.ndarray:
        """
        Encode an assignment into a unit vector [0,1]^D following param order.
        """
        values: List[float] = []
        for p in self.params:
            values.append(float(p.to_unit(assignment[p.name])))
        for block in self._active_conditionals(assignment):
            for p in block.params:
                values.append(float(p.to_unit(assignment[p.name])))
        return np.asarray(values, dtype=float)

    def from_unit_vector(self, u: np.ndarray) -> Dict[str, Any]:
        """
        Decode a unit vector into an assignment dict, respecting conditionals.
        """
        assignment: Dict[str, Any] = {}
        idx = 0
        for p in self.params:
            assignment[p.name] = p.from_unit(float(u[idx]))
            idx += 1
        for block in self._active_conditionals(assignment):
            for p in block.params:
                assignment[p.name] = p.from_unit(float(u[idx]))
                idx += 1
        return assignment
