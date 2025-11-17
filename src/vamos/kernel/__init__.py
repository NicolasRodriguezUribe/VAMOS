"""Kernel implementations for multi-objective optimization.

This module provides different backends for the core operations of the
evolutionary algorithms, including NumPy, Numba, and moocore backends.
"""

from .numpy_backend import NumPyKernel

__all__ = [
    "NumPyKernel",
]
