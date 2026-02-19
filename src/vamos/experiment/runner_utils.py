"""
Helper utilities shared by runner components.
"""

from __future__ import annotations

import os

import numpy as np

from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.problem.types import ProblemProtocol
from vamos.foundation.problem.registry import ProblemSelection
from vamos.foundation.core.experiment_config import ExperimentConfig


def validate_problem(problem: ProblemProtocol) -> None:
    if problem.n_var <= 0 or problem.n_obj <= 0:
        raise ValueError("Problem must have positive n_var and n_obj.")
    if problem.n_constraints < 0:
        raise ValueError("n_constraints must be >= 0.")
    xl = np.asarray(problem.xl)
    xu = np.asarray(problem.xu)
    if xl.ndim > 1 or xu.ndim > 1:
        raise ValueError("Problem bounds must be scalars or 1D arrays.")
    if xl.ndim == 1 and xl.shape[0] != problem.n_var:
        raise ValueError("Lower bounds length must match n_var.")
    if xu.ndim == 1 and xu.shape[0] != problem.n_var:
        raise ValueError("Upper bounds length must match n_var.")
    if np.any(xl > xu):
        raise ValueError("Lower bounds must not exceed upper bounds.")
    encoding = normalize_encoding(problem.encoding)
    if encoding == "mixed":
        if not hasattr(problem, "mixed_spec"):
            raise ValueError("Mixed-encoding problems must define 'mixed_spec'.")
        spec = getattr(problem, "mixed_spec")
        spec_keys = set(spec.keys())
        if not {"perm_idx", "real_idx", "int_idx", "cat_idx"} & spec_keys:
            raise ValueError("mixed_spec must define at least one of perm_idx, real_idx, int_idx, or cat_idx.")
        if "real_idx" in spec_keys and not {"real_lower", "real_upper"} <= spec_keys:
            raise ValueError("mixed_spec missing required fields: real_lower, real_upper.")
        if "int_idx" in spec_keys and not {"int_lower", "int_upper"} <= spec_keys:
            raise ValueError("mixed_spec missing required fields: int_lower, int_upper.")
        if "cat_idx" in spec_keys and "cat_cardinality" not in spec_keys:
            raise ValueError("mixed_spec missing required field: cat_cardinality.")


def problem_output_dir(selection: ProblemSelection, config: ExperimentConfig) -> str:
    spec = getattr(selection, "spec", None)
    label = getattr(spec, "label", None) or getattr(spec, "key", "unknown")
    safe = str(label).replace(" ", "_").upper()
    return os.path.join(config.output_root, f"{safe}")


def run_output_dir(selection: ProblemSelection, algorithm_name: str, engine_name: str, seed: int, config: ExperimentConfig) -> str:
    base = problem_output_dir(selection, config)
    return os.path.join(
        base,
        algorithm_name.lower(),
        engine_name.lower(),
        f"seed_{seed}",
    )


__all__ = ["validate_problem", "problem_output_dir", "run_output_dir"]
