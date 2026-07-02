from __future__ import annotations

import numpy as np
import pytest

from vamos.engine.hyperheuristics.indicator import IndicatorEvaluator


def test_hypervolume_indicator_requires_reference_point() -> None:
    evaluator = IndicatorEvaluator("hv")

    with pytest.raises(ValueError, match="reference_point"):
        evaluator.compute(np.array([[0.2, 0.2]], dtype=float))


def test_hypervolume_indicator_applies_mode() -> None:
    F = np.array([[0.2, 0.2]], dtype=float)

    maximize = IndicatorEvaluator("hv", reference_point=np.array([1.0, 1.0]), mode="maximize")
    minimize = IndicatorEvaluator("hv", reference_point=np.array([1.0, 1.0]), mode="minimize")

    assert maximize.compute(F) > 0.0
    assert minimize.compute(F) == pytest.approx(-maximize.compute(F))


def test_igd_indicator_requires_reference_front_before_optional_backend() -> None:
    with pytest.raises(ValueError, match="reference_front"):
        IndicatorEvaluator("igd")


def test_igd_indicator_works_with_reference_front_when_moocore_available() -> None:
    pytest.importorskip("moocore")
    front = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

    evaluator = IndicatorEvaluator("igd", reference_front=front, mode="minimize")

    assert evaluator.compute(front) <= 0.0
