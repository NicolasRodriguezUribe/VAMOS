import numpy as np
import pytest

from vamos.foundation.metrics import moocore_indicators as mi


@pytest.mark.skipif(not mi.has_moocore(), reason="MooCore not installed")
def test_hv_wrapper_matches_moocore():
    from moocore import _moocore as mc  # type: ignore

    F = np.array([[0.2, 0.3], [0.4, 0.1]])
    ref = np.array([1.0, 1.0])

    wrapper = mi.HVIndicator(reference_point=ref)
    res = wrapper.compute(F)

    expected = mc.hypervolume(F, ref=ref)
    assert np.isclose(res.value, expected)


@pytest.mark.skipif(not mi.has_moocore(), reason="MooCore not installed")
def test_igd_and_epsilon_are_zero_on_reference():
    ref = np.array([[1.0, 1.0], [2.0, 2.0]])
    approx = ref.copy()

    igd = mi.IGDIndicator(reference_front=ref).compute(approx).value
    igd_plus = mi.IGDPlusIndicator(reference_front=ref).compute(approx).value
    eps_add = mi.EpsilonAdditiveIndicator(reference_front=ref).compute(approx).value
    eps_mult = mi.EpsilonMultiplicativeIndicator(reference_front=ref).compute(approx).value

    assert igd == pytest.approx(0.0)
    assert igd_plus == pytest.approx(0.0)
    assert eps_add == pytest.approx(0.0)
    assert eps_mult == pytest.approx(1.0)
