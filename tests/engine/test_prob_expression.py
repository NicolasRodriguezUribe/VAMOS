"""Tests for probability expression parsing."""

from __future__ import annotations

from vamos.engine.algorithm.components.utils import resolve_prob_expression


def test_resolve_prob_expression_parses_division() -> None:
    assert resolve_prob_expression("1/n", 10) == 0.1


def test_resolve_prob_expression_clamps_bounds() -> None:
    assert resolve_prob_expression(2.0, 10) == 1.0
    assert resolve_prob_expression(-0.5, 10) == 0.0


def test_resolve_prob_expression_handles_defaults() -> None:
    assert resolve_prob_expression(None, 10, default=0.2) == 0.2
    assert resolve_prob_expression("0.25", 10, default=0.2) == 0.25
    assert resolve_prob_expression("bogus", 10, default=0.3) == 0.3
