# algorithm/nsgaii_operators.py
"""
Backward-compatibility shim for nsgaii_operators module.

The implementation has moved to vamos.algorithm.nsgaii.operators.
This module re-exports the public API for backward compatibility.
"""
from vamos.algorithm.nsgaii.operators import build_operator_pool

__all__ = ["build_operator_pool"]
