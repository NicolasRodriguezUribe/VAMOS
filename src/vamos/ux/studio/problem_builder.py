"""Compatibility wrapper for the Studio problem builder page."""

from __future__ import annotations

from vamos.ux.studio.problem_builder_backend import (
    _DEFAULT_TEMPLATE,
    example_objectives,
    generate_script,
    parse_bounds_text,
)
from vamos.ux.studio.problem_builder_page import render_problem_builder

_example_objectives = example_objectives
_parse_bounds_text = parse_bounds_text
_generate_script = generate_script

__all__ = ["render_problem_builder"]
