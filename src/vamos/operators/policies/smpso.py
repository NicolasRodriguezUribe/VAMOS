"""SMPSO operator building.

This module provides factory functions for building SMPSO operators:
- Mutation operator (polynomial mutation / turbulence)
- Repair operator for bounds handling
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos.operators.impl.real import PolynomialMutation
from vamos.engine.algorithm.components.variation import prepare_mutation_params
from vamos.operators.impl.real import VariationWorkspace
from vamos.engine.algorithm.smpso.helpers import resolve_repair
from vamos.foundation.encoding import normalize_encoding


__all__ = [
    "build_mutation_operator",
    "build_repair_operator",
]


def build_mutation_operator(
    config: dict[str, Any],
    encoding: str,
    n_var: int,
    xl: np.ndarray,
    xu: np.ndarray,
) -> Any:
    """Build the mutation (turbulence) operator for SMPSO.

    SMPSO uses polynomial mutation to add turbulence to particle positions,
    helping to escape local optima.

    Parameters
    ----------
    config : dict
        Algorithm configuration with 'mutation' key.
    encoding : str
        Variable encoding (must be continuous/real for SMPSO).
    n_var : int
        Number of decision variables.
    xl : np.ndarray
        Lower bounds.
    xu : np.ndarray
        Upper bounds.

    Returns
    -------
    PolynomialMutation
        Configured mutation operator.

    Raises
    ------
    ValueError
        If mutation type is not supported.
    """
    mut_method, mut_params = config.get("mutation", ("pm", {}))
    mut_method = str(mut_method).lower()
    if mut_method not in {"pm", "polynomial"}:
        raise ValueError(f"Unsupported SMPSO mutation '{mut_method}'.")

    normalized = normalize_encoding(encoding)
    mut_params = prepare_mutation_params(dict(mut_params or {}), normalized, n_var)
    workspace = VariationWorkspace()
    prob_raw = mut_params.get("prob")
    prob = float(prob_raw) if prob_raw is not None else 1.0 / max(1, n_var)
    eta_raw = mut_params.get("eta")
    eta = float(eta_raw) if eta_raw is not None else 20.0

    return PolynomialMutation(
        prob_mutation=prob,
        eta=eta,
        lower=xl,
        upper=xu,
        workspace=workspace,
    )


def build_repair_operator(config: dict[str, Any]) -> Any | None:
    """Build the repair operator from configuration.

    Parameters
    ----------
    config : dict
        Algorithm configuration with optional 'repair' key.

    Returns
    -------
    Any or None
        Repair operator instance or None if not configured.
    """
    return resolve_repair(config.get("repair"))
