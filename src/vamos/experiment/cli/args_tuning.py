from __future__ import annotations

import argparse

from .types import SpecDefaults


def add_tuning_arguments(
    parser: argparse.ArgumentParser,
    *,
    spec_defaults: SpecDefaults,
) -> None:
    """Register tuning-related arguments (none defined yet)."""
    _ = parser
    _ = spec_defaults
