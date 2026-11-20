import numpy as np
import pytest

from conftest import synthetic_signal
from reference import notebook_gain_array, notebook_preset_gain, notebook_trend
from seismic_signal_attributes.characteristic import spline_scale
from seismic_signal_attributes.gain import (
    amplitude_recovery,
    amplitude_trend,
    fit_gain,
    gain_curve,
    preset_gain,
)

ONSET = 2998


@pytest.fixture(scope="module")
def signal():
    return spline_scale(synthetic_signal(0))


def test_preset_gain_matches_notebook(signal):
    gained, gain = preset_gain(signal, ONSET)
    np.testing.assert_allclose(gained, notebook_preset_gain(ONSET, signal), rtol=1e-12)
    assert gain[ONSET] == 1.0 and gain[ONSET + 199] == 1.0
    assert gain[ONSET + 200] == pytest.approx(1.0)  # continuous at the end of the hold
    assert gain[-1] < 2.0 and gain[0] < 1e-10


def test_amplitude_trend_matches_notebook(signal):
    np.testing.assert_allclose(amplitude_trend(signal, ONSET), notebook_trend(signal, ONSET), atol=1e-12)


def test_amplitude_trend_is_block_peak_envelope():
    signal = np.zeros(100)
    signal[45] = -0.8
    trend = amplitude_trend(signal, onset=40, step=20)
    assert np.all(trend[:40] == 1.0)
    assert trend[45] == pytest.approx(0.8)
    assert trend[45:].max() == pytest.approx(0.8)


def test_gain_curve_matches_notebook(signal):
    params = (0.85, 0.011, 0.05)
    expected_gain, expected_rec = notebook_gain_array(*params, ONSET, signal)
    gain = gain_curve(params, len(signal), ONSET)
    np.testing.assert_allclose(gain, expected_gain, rtol=1e-12)
    np.testing.assert_allclose(signal / gain, expected_rec, rtol=1e-12)


def test_fit_recovers_known_parameters():
    n, onset = 4000, 500
    trend = np.ones(n)
    trend[onset:] = 0.9 * np.exp(-0.01 * np.arange(n - onset)) + 0.05
    a, b, c = fit_gain(trend, onset)
    assert (a, b, c) == pytest.approx((0.9, 0.01, 0.05), rel=1e-4)


def test_amplitude_recovery_flattens_decay(signal):
    recovered, gain, (a, b, c) = amplitude_recovery(signal, ONSET)
    assert b > 0 and gain.max() == pytest.approx(1.0)
    assert np.all(gain[:ONSET] == 1.0)
    late = slice(ONSET + 400, ONSET + 600)
    assert np.abs(recovered[late]).max() > 5 * np.abs(signal[late]).max()


def test_non_positive_gain_is_rejected():
    with pytest.raises(ValueError, match="not positive"):
        gain_curve((1.0, 0.01, -0.5), 1000, 100)


@pytest.mark.parametrize("onset", [-1, 6000])
def test_onset_outside_trace(signal, onset):
    with pytest.raises(ValueError, match="outside the trace"):
        preset_gain(signal, onset)
