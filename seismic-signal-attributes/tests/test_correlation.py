import numpy as np
import pytest

from conftest import synthetic_signal
from reference import notebook_impulse_corr, notebook_xcorr_sum
from seismic_signal_attributes.characteristic import spline_scale
from seismic_signal_attributes.correlation import (
    _shift_left,
    aligned_correlation,
    aligned_correlation_sum,
    impulse,
    impulse_correlation,
)

N, ONSET = 3000, 1400


@pytest.fixture(scope="module")
def signals():
    return [spline_scale(synthetic_signal(i, n=N, arrival=1400 + 20 * i)) for i in range(3)]


def test_cross_correlation_sum_matches_notebook(signals):
    a, others = signals[0], signals[1:]
    np.testing.assert_allclose(
        aligned_correlation_sum(a, others, ONSET), notebook_xcorr_sum(a, others, ONSET), atol=1e-9
    )


def test_auto_plus_cross_sum_matches_notebook(signals):
    a, others = signals[0], signals[1:]
    ours = aligned_correlation_sum(a, others, ONSET) + aligned_correlation(a, a, ONSET)
    np.testing.assert_allclose(ours, notebook_xcorr_sum(a, others, ONSET, include_self=True), atol=1e-9)


def test_peak_is_moved_to_onset(signals):
    xc = aligned_correlation(signals[0], signals[1], ONSET)
    assert len(xc) == N
    assert np.argmax(np.abs(xc)) == ONSET


def test_shift_does_not_wrap():
    x = np.arange(1.0, 6.0)
    np.testing.assert_array_equal(_shift_left(x, 2), [3, 4, 5, 0, 0])
    np.testing.assert_array_equal(_shift_left(x, -2), [0, 0, 1, 2, 3])


def test_impulse_correlation_matches_notebook(signals):
    a, b = signals[0], signals[1]
    pulse = impulse(b, 1420)
    np.testing.assert_allclose(
        impulse_correlation(a, ONSET, pulse), notebook_impulse_corr(a, ONSET, pulse), atol=1e-12
    )


def test_impulse_autocorrelation_peaks_at_onset(signals):
    a = signals[0]
    out = impulse_correlation(a, ONSET, impulse(a, ONSET))
    assert np.all(out[-60:] == 0) and np.all(out[: ONSET - 60] == 0)
    assert out[ONSET] == pytest.approx(np.sum(a[ONSET : ONSET + 60] ** 2))


def test_impulse_correlation_does_not_modify_input(signals):
    a = signals[0].copy()
    impulse_correlation(a, ONSET, impulse(a, ONSET))
    np.testing.assert_array_equal(a, signals[0])


def test_impulse_must_fit():
    with pytest.raises(ValueError, match="does not fit"):
        impulse(np.zeros(100), 50)
