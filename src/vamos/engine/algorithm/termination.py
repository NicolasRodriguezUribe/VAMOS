# algorithm/termination.py
"""
Backward-compatibility shim for termination module.

The implementation has moved to vamos.engine.algorithm.components.termination.
This module re-exports the public API for backward compatibility.
"""
from vamos.engine.algorithm.components.termination import HVTracker

__all__ = ["HVTracker"]
