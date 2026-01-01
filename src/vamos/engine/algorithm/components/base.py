"""
Shared algorithm component exports.

This module keeps a thin public surface while delegating implementation
to focused component modules.
"""

from __future__ import annotations

from vamos.engine.algorithm.components.archives import (
    resolve_archive_size,
    setup_archive,
    update_archive,
)
from vamos.engine.algorithm.components.hooks import (
    finalize_genealogy,
    get_live_viz,
    live_should_stop,
    match_ids,
    notify_generation,
    setup_genealogy,
    track_offspring_genealogy,
)
from vamos.engine.algorithm.components.lifecycle import (
    get_eval_backend,
    setup_initial_population,
)
from vamos.engine.algorithm.components.metrics import setup_hv_tracker
from vamos.engine.algorithm.components.results import build_result
from vamos.engine.algorithm.components.state import AlgorithmState
from vamos.engine.algorithm.components.termination import parse_termination

__all__ = [
    # State
    "AlgorithmState",
    # Termination
    "parse_termination",
    "setup_hv_tracker",
    # Population lifecycle
    "setup_initial_population",
    "get_eval_backend",
    # Archives
    "setup_archive",
    "update_archive",
    "resolve_archive_size",
    # Live visualization hooks
    "get_live_viz",
    "notify_generation",
    "live_should_stop",
    # Genealogy
    "setup_genealogy",
    "track_offspring_genealogy",
    "finalize_genealogy",
    "match_ids",
    # Results
    "build_result",
]
