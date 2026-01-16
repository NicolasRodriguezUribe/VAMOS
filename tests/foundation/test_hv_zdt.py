import numpy as np

from vamos.foundation.metrics.hv_zdt import compute_normalized_hv


def test_normalized_hv_handles_points_above_default_reference():
    # Points exceed the default [1.1, 1.1] reference; function should auto-expand ref.
    F = np.array([[1.2, 1.0], [1.05, 1.3]])
    hv = compute_normalized_hv(F, "zdt1")
    assert hv >= 0.0
    assert hv <= 1.0 + 1e-6


def test_normalized_hv_zdt5_reference_front_is_one():
    from vamos.foundation.metrics.hv_zdt import get_zdt_reference_front

    pf = get_zdt_reference_front("zdt5")
    hv = compute_normalized_hv(pf, "zdt5")
    assert np.isclose(hv, 1.0)
