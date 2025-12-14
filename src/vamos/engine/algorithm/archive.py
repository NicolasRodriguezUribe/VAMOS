# algorithm/archive.py
"""
Backward-compatibility shim for archive module.

The implementation has moved to vamos.engine.algorithm.components.archive.
This module re-exports the public API for backward compatibility.
"""
from vamos.engine.algorithm.components.archive import (
    CrowdingDistanceArchive,
    HypervolumeArchive,
    _single_front_crowding,
)

__all__ = [
    "CrowdingDistanceArchive",
    "HypervolumeArchive",
    "_single_front_crowding",
]
