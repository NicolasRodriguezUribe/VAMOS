import numpy as np
import pytest

from vamos.foundation.metrics import moocore_indicators as mi

pytestmark = pytest.mark.backends


@pytest.mark.skipif(not mi.has_moocore(), reason="MooCore not installed")
def test_hv_wrapper_matches_moocore():
    import moocore

    F = np.array([[0.2, 0.3], [0.4, 0.1]])
    ref = np.array([1.0, 1.0])

    wrapper = mi.HVIndicator(reference_point=ref)
    res = wrapper.compute(F)

    expected = moocore.hypervolume(F, ref=ref)
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


@pytest.mark.skipif(not mi.has_moocore(), reason="MooCore not installed")
def test_indicators_match_moocore_reference_implementation():
    import moocore

    approx = np.array([[3.5, 5.5], [3.6, 4.1], [4.1, 3.2], [5.5, 1.5]])
    ref_front = np.array([[1, 6], [2, 5], [3, 4], [4, 3], [5, 2], [6, 1]], dtype=float)

    assert mi.IGDIndicator(reference_front=ref_front).compute(approx).value == pytest.approx(moocore.igd(approx, ref=ref_front))
    assert mi.IGDPlusIndicator(reference_front=ref_front).compute(approx).value == pytest.approx(moocore.igd_plus(approx, ref=ref_front))
    assert mi.EpsilonAdditiveIndicator(reference_front=ref_front).compute(approx).value == pytest.approx(
        moocore.epsilon_additive(approx, ref=ref_front)
    )
    assert mi.EpsilonMultiplicativeIndicator(reference_front=ref_front).compute(approx).value == pytest.approx(
        moocore.epsilon_mult(approx, ref=ref_front)
    )
    assert mi.AvgHausdorffIndicator(reference_front=ref_front).compute(approx).value == pytest.approx(
        moocore.avg_hausdorff_dist(approx, ref_front)
    )


@pytest.mark.skipif(not mi.has_moocore(), reason="MooCore not installed")
def test_hv_wrapper_supports_maximise():
    import moocore

    F = np.array([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]])
    ref = np.array([0.0, 0.0])

    wrapper = mi.HVIndicator(reference_point=ref)
    res = wrapper.compute(F, maximise=True)
    assert res.value == pytest.approx(moocore.hypervolume(F, ref=ref, maximise=True))
