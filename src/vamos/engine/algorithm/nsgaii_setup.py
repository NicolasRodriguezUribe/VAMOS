# algorithm/nsgaii_setup.py
"""
Backward-compatibility shim for nsgaii_setup module.

The implementation has moved to vamos.engine.algorithm.nsgaii.setup.
This module re-exports the public API for backward compatibility.
"""
from vamos.engine.algorithm.nsgaii.setup import (
    parse_termination,
    setup_population,
    setup_archive,
    setup_genealogy,
    setup_selection,
    setup_result_archive,
    build_operator_pool,
    resolve_archive_size,
    DEFAULT_TOURNAMENT_PRESSURE,
)

__all__ = [
    "parse_termination",
    "setup_population",
    "setup_archive",
    "setup_genealogy",
    "setup_selection",
    "setup_result_archive",
    "build_operator_pool",
    "resolve_archive_size",
    "DEFAULT_TOURNAMENT_PRESSURE",
]
