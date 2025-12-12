# algorithm/hypervolume.py
"""
Backward-compatibility shim for hypervolume module.

The implementation has moved to vamos.algorithm.components.hypervolume.
This module re-exports the public API for backward compatibility.
"""
from vamos.algorithm.components.hypervolume import (
    hypervolume,
    hypervolume_contributions,
    _hypervolume_impl,
    _hypervolume_contributions_2d,
)

__all__ = [
    "hypervolume",
    "hypervolume_contributions",
    "_hypervolume_impl",
    "_hypervolume_contributions_2d",
]
