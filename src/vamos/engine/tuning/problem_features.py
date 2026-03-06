"""
Problem feature extraction for knowledge-based MOEA tuning.

Computes a feature vector from problem metadata that characterizes the
optimization task. Used by the KnowledgeBase to find similar problems
and warm-start the tuner with proven configurations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProblemFeatures:
    """Feature vector describing an optimization problem.

    Attributes:
        n_var: Number of decision variables.
        n_obj: Number of objectives.
        n_constraints: Number of constraints.
        encoding: Variable encoding type.
        var_range_ratio: Ratio of max to min variable range (scale heterogeneity).
        problem_family: Problem suite name if known (zdt, dtlz, wfg, custom).
    """

    n_var: int
    n_obj: int
    n_constraints: int = 0
    encoding: str = "real"
    var_range_ratio: float = 1.0
    problem_family: str = "custom"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProblemFeatures:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def distance(self, other: ProblemFeatures) -> float:
        """Compute a weighted distance between two feature vectors.

        Continuous features are normalized; categorical features contribute
        a binary mismatch penalty.
        """
        d = 0.0
        # Normalized numeric distances
        d += abs(self.n_var - other.n_var) / max(self.n_var, other.n_var, 1)
        d += abs(self.n_obj - other.n_obj) / max(self.n_obj, other.n_obj, 1)
        d += abs(self.n_constraints - other.n_constraints) / max(self.n_constraints, other.n_constraints, 1)
        d += abs(self.var_range_ratio - other.var_range_ratio) / max(self.var_range_ratio, other.var_range_ratio, 1.0)
        # Categorical mismatches
        if self.encoding != other.encoding:
            d += 1.0
        if self.problem_family != other.problem_family:
            d += 0.5
        return d


def extract_features_from_problem(problem: Any) -> ProblemFeatures:
    """Extract features from a VAMOS problem object.

    Works with any object following the ProblemProtocol (n_var, n_obj, etc.).
    """
    n_var = getattr(problem, "n_var", 1)
    n_obj = getattr(problem, "n_obj", 2)
    n_constraints = getattr(problem, "n_constraints", 0)
    encoding = str(getattr(problem, "encoding", "real"))

    # Compute variable range ratio from bounds if available
    xl = getattr(problem, "xl", None)
    xu = getattr(problem, "xu", None)
    var_range_ratio = 1.0
    if xl is not None and xu is not None:
        xl_arr = np.asarray(xl, dtype=float)
        xu_arr = np.asarray(xu, dtype=float)
        ranges = xu_arr - xl_arr
        positive = ranges[ranges > 0]
        if len(positive) >= 2:
            var_range_ratio = float(np.max(positive) / np.min(positive))

    # Try to infer problem family from name
    name = str(getattr(problem, "name", getattr(problem, "__class__", "")))
    family = _infer_family(name.lower())

    return ProblemFeatures(
        n_var=int(n_var),
        n_obj=int(n_obj),
        n_constraints=int(n_constraints),
        encoding=encoding,
        var_range_ratio=var_range_ratio,
        problem_family=family,
    )


def extract_features_from_instances(
    instances: list[Any],
    *,
    default_n_obj: int = 2,
) -> ProblemFeatures:
    """Extract aggregate features from a list of tuning instances.

    Takes the most common / average values across the instance list.
    """
    if not instances:
        return ProblemFeatures(n_var=10, n_obj=default_n_obj)

    n_vars = []
    n_objs = []
    families = []
    for inst in instances:
        name = str(getattr(inst, "name", ""))
        n_var = getattr(inst, "n_var", None)
        kwargs = getattr(inst, "kwargs", {}) or {}
        if n_var is None:
            n_var = kwargs.get("n_var", 10)
        n_obj = kwargs.get("n_obj", default_n_obj)
        n_vars.append(int(n_var))
        n_objs.append(int(n_obj))
        families.append(_infer_family(name.lower()))

    # Use median n_var and most common n_obj
    median_nvar = int(np.median(n_vars))
    # Most common n_obj
    unique, counts = np.unique(n_objs, return_counts=True)
    common_nobj = int(unique[np.argmax(counts)])
    # Most common family
    from collections import Counter

    family_counts = Counter(families)
    common_family = family_counts.most_common(1)[0][0]

    return ProblemFeatures(
        n_var=median_nvar,
        n_obj=common_nobj,
        problem_family=common_family,
    )


def _infer_family(name: str) -> str:
    """Infer problem family from name string."""
    for prefix in ("zdt", "dtlz", "wfg", "cec", "uf", "cf"):
        if prefix in name:
            return prefix
    return "custom"


__all__ = [
    "ProblemFeatures",
    "extract_features_from_problem",
    "extract_features_from_instances",
]
