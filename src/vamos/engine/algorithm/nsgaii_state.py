# algorithm/nsgaii_state.py
"""
Backward-compatibility shim for nsgaii_state module.

The implementation has moved to vamos.engine.algorithm.nsgaii.state.
This module re-exports the public API for backward compatibility.
"""
from vamos.engine.algorithm.nsgaii.state import (
    NSGAIIState,
    build_result,
    finalize_genealogy,
    compute_selection_metrics,
    track_offspring_genealogy,
    update_archives,
)

__all__ = [
    "NSGAIIState",
    "build_result",
    "finalize_genealogy",
    "compute_selection_metrics",
    "track_offspring_genealogy",
    "update_archives",
]
