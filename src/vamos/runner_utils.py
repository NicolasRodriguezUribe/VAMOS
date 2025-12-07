"""
Helper utilities shared by runner components.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from vamos.problem.types import ProblemProtocol, MixedProblemProtocol
from vamos.problem.resolver import ProblemSelection
from vamos.experiment_config import ExperimentConfig


def validate_problem(problem: ProblemProtocol) -> None:
    if problem.n_var <= 0 or problem.n_obj <= 0:
        raise ValueError("Problem must have positive n_var and n_obj.")
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
    encoding = getattr(problem, "encoding", "continuous")
    if encoding == "mixed":
        if not hasattr(problem, "mixed_spec"):
            raise ValueError("Mixed-encoding problems must define 'mixed_spec'.")
        spec = getattr(problem, "mixed_spec")
        required = {"real_idx", "int_idx", "cat_idx", "real_lower", "real_upper", "int_lower", "int_upper", "cat_cardinality"}
        missing = required - set(spec.keys())
        if missing:
            raise ValueError(f"mixed_spec missing required fields: {', '.join(sorted(missing))}")


def problem_output_dir(selection: ProblemSelection, config: ExperimentConfig) -> str:
    spec = getattr(selection, "spec", None)
    label = getattr(spec, "label", None) or getattr(spec, "key", "unknown")
    safe = str(label).replace(" ", "_").upper()
    return os.path.join(config.output_root, f"{safe}")


def run_output_dir(
    selection: ProblemSelection, algorithm_name: str, engine_name: str, seed: int, config: ExperimentConfig
) -> str:
    base = problem_output_dir(selection, config)
    return os.path.join(
        base,
        algorithm_name.lower(),
        engine_name.lower(),
        f"seed_{seed}",
    )


__all__ = ["validate_problem", "problem_output_dir", "run_output_dir"]
