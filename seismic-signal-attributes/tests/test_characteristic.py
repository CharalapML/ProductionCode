import numpy as np
import pytest

from conftest import synthetic_signal
from reference import notebook_cfs
from seismic_signal_attributes.characteristic import (
    characteristic_functions,
    kurtosis_cf,
    spline_scale,
)


@pytest.fixture(scope="module")
def signal():
    return spline_scale(synthetic_signal(0, n=2500, arrival=1200))


@pytest.fixture(scope="module")
def reference(signal):
    return notebook_cfs(signal)


@pytest.mark.parametrize("name", ["F1cf", "F2cf", "F3cf", "F4cf", "F4cfgrad_gtmean", "env_pos"])
def test_matches_notebook(signal, reference, name):
    np.testing.assert_allclose(characteristic_functions(signal)[name], reference[name], rtol=1e-9, atol=1e-12)


def test_gradient_and_threshold_match_notebook(signal, reference):
    cf = characteristic_functions(signal)
    # the notebook leaves the last gradient sample unclipped; the package clips it too
    np.testing.assert_allclose(cf["F4cfgrad"][:-1], reference["F4cfgrad"][:-1], rtol=1e-9, atol=1e-12)
    assert cf["F4cfgrad"][-1] >= 0
    assert cf["threshold"] == pytest.approx(reference["threshold"], rel=1e-9)


def test_spline_scale_normalises_and_does_not_modify_input():
    raw = synthetic_signal(1) * 1e4
    before = raw.copy()
    scaled = spline_scale(raw)
    assert np.abs(scaled).max() == pytest.approx(1.0)
    np.testing.assert_array_equal(raw, before)


def test_spline_scale_accepts_integer_data():
    scaled = spline_scale((synthetic_signal(2) * 1e4).astype(np.int32))
    assert scaled.dtype == np.float64


@pytest.mark.parametrize(
    "data, match",
    [(np.zeros(5000), "flat"), (np.ones(100), "at least"), (np.ones((2, 3000)), "1-D")],
)
def test_spline_scale_rejects_bad_input(data, match):
    with pytest.raises(ValueError, match=match):
        spline_scale(data)


def test_kurtosis_of_gaussian_noise_is_about_three_quarters():
    noise = np.random.default_rng(0).normal(size=20000)
    f1 = kurtosis_cf(noise)
    assert np.all(f1[:198] == 0)
    assert f1[198:].mean() == pytest.approx(3 / 4, abs=0.05)  # kurtosis 3, divided by 4


def test_kurtosis_of_constant_signal_is_zero_not_nan():
    assert np.all(kurtosis_cf(np.ones(500)) == 0)


def test_kurtosis_rejects_short_signal():
    with pytest.raises(ValueError, match="at least 200"):
        kurtosis_cf(np.ones(150))


def test_no_positive_gradient_gives_zero_threshold():
    cf = characteristic_functions(np.ones(500))
    assert cf["threshold"] == 0.0
    assert not cf["F4cfgrad_gtmean"].any()
