from __future__ import annotations

import numpy as np

from vamos.foundation.metrics.hv_zdt import compute_normalized_hv, get_zdt_reference_front
from vamos.foundation.metrics.hypervolume import compute_hypervolume


def test_compute_hypervolume_returns_expected_area() -> None:
    F = np.array([[0.5, 1.5], [1.5, 0.5]])
    hv = compute_hypervolume(F, ref_point=[2.0, 2.0])
    assert np.isclose(hv, 1.25)


def test_compute_hypervolume_returns_zero_when_outside_ref() -> None:
    F = np.array([[3.0, 3.0], [4.0, 2.5]])
    hv = compute_hypervolume(F, ref_point=[2.0, 2.0])
    assert hv == 0.0


def test_compute_normalized_hv_matches_reference_front() -> None:
    front = get_zdt_reference_front("zdt1")
    hv_norm = compute_normalized_hv(front, "zdt1")
    assert np.isclose(hv_norm, 1.0)


def test_compute_normalized_hv_returns_zero_for_empty_front() -> None:
    hv_norm = compute_normalized_hv(np.empty((0, 2)), "zdt1")
    assert hv_norm == 0.0
