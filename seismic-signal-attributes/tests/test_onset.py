import numpy as np
import pytest

from conftest import synthetic_signal
from reference import notebook_onset, notebook_subsequent_onsets
from seismic_signal_attributes.characteristic import characteristic_functions, spline_scale
from seismic_signal_attributes.onset import (
    OnsetNotFoundError,
    detect_onset,
    detect_subsequent_onsets,
)


@pytest.fixture(scope="module", params=[0, 1, 2, 3])
def cf(request):
    return characteristic_functions(spline_scale(synthetic_signal(request.param)))


def _args(cf):
    return cf["F1cf"], cf["F4cf"], cf["F4cfgrad_gtmean"], cf["threshold"]


def test_onset_matches_notebook(cf):
    expected = notebook_onset(cf["threshold"], cf["F4cfgrad_gtmean"], cf["F1cf"], cf["F4cf"])
    assert detect_onset(*_args(cf)) == expected


def test_onset_is_close_to_true_arrival(cf):
    assert abs(detect_onset(*_args(cf)) - 3000) <= 5


def test_subsequent_onsets_match_notebook(cf):
    expected = notebook_subsequent_onsets(cf["F1cf"], cf["threshold"], cf["F4cfgrad_gtmean"], cf["F4cf"])
    np.testing.assert_array_equal(detect_subsequent_onsets(*_args(cf)), expected)


def test_handcrafted_onset():
    f1 = np.zeros(20)
    f1[10] = 1.0
    f4 = np.ones(20)
    f4[6] = 0.0
    gtmean = np.zeros(20)
    gtmean[9] = 5.0
    # x=9 exceeds the threshold; nearest zero of F4cf behind it is at 6
    assert detect_onset(f1, f4, gtmean, threshold=1.0) == 6


def test_onset_not_found():
    zeros = np.zeros(50)
    with pytest.raises(OnsetNotFoundError, match="no onset"):
        detect_onset(zeros, zeros, zeros, threshold=0.0)


def test_onset_never_wraps_to_end_of_trace():
    f1 = np.zeros(20)
    f1[1] = 1.0
    f4 = np.ones(20)
    f4[-1] = 0.0  # a negative index would find this
    gtmean = np.full(20, 5.0)
    with pytest.raises(OnsetNotFoundError):
        detect_onset(f1, f4, gtmean, threshold=1.0)
