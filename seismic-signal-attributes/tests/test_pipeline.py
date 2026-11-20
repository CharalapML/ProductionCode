import copy

import numpy as np
import pytest

from conftest import ARRIVALS, N_SAMPLES, synthetic_signal
from seismic_signal_attributes.characteristic import spline_scale
from seismic_signal_attributes.correlation import aligned_correlation
from seismic_signal_attributes.io import Trace
from seismic_signal_attributes.pipeline import (
    ATTRIBUTE_NAMES,
    TraceProcessingError,
    build_attribute_array,
)


@pytest.fixture(scope="module")
def built(request):
    traces = request.getfixturevalue("traces")
    originals = copy.deepcopy(traces)
    array, metadata = build_attribute_array(traces)
    return traces, originals, array, metadata


@pytest.fixture(scope="module")
def traces():
    return [Trace(f"R{i}", synthetic_signal(i, arrival=a), 20.0) for i, a in enumerate(ARRIVALS)]


def test_shape_and_finite(built):
    _, _, array, metadata = built
    assert array.shape == (3, len(ATTRIBUTE_NAMES), N_SAMPLES) == (3, 12, N_SAMPLES)
    assert np.all(np.isfinite(array))
    assert len(metadata) == 3


def test_onsets_near_true_arrivals(built):
    _, _, _, metadata = built
    for meta, arrival in zip(metadata, ARRIVALS):
        assert abs(meta["onset_sample"] - arrival) <= 5
        assert meta["onset_time_s"] == pytest.approx(meta["onset_sample"] / 20.0)
        assert all(s > meta["onset_sample"] for s in meta["subsequent_onset_samples"])


def test_first_attribute_is_scaled_signal(built):
    traces, _, array, _ = built
    np.testing.assert_allclose(array[1, 0], spline_scale(traces[1].data))


def test_axcorr_is_xcorr_plus_autocorrelation(built):
    _, _, array, metadata = built
    x, ax = ATTRIBUTE_NAMES.index("XcorrShiftAdded"), ATTRIBUTE_NAMES.index("AXcorrShiftAdded")
    s, onset = array[0, 0], metadata[0]["onset_sample"]
    np.testing.assert_allclose(array[0, ax] - array[0, x], aligned_correlation(s, s, onset), atol=1e-9)


def test_inputs_are_not_modified(built):
    traces, originals, _, _ = built
    for t, o in zip(traces, originals):
        np.testing.assert_array_equal(t.data, o.data)


def test_results_do_not_depend_on_trace_order(built, traces):
    _, _, array, _ = built
    reversed_array, _ = build_attribute_array(traces[::-1])
    np.testing.assert_allclose(reversed_array[::-1], array, atol=1e-9)


def test_single_trace_has_zero_cross_correlations():
    array, _ = build_attribute_array([Trace("R0", synthetic_signal(0), 20.0)])
    for name in ("XcorrShiftAdded", "AllImpXcorrAdded_additional"):
        assert not array[0, ATTRIBUTE_NAMES.index(name)].any()


def test_shorter_traces_are_zero_padded():
    traces = [Trace("long", synthetic_signal(0), 20.0), Trace("short", synthetic_signal(1, n=5000), 20.0)]
    array, metadata = build_attribute_array(traces)
    assert array.shape == (2, 12, N_SAMPLES)
    assert metadata[1]["npts"] == 5000
    assert not array[1, :, 5000:].any()


def test_error_names_the_trace():
    traces = [Trace("good", synthetic_signal(0), 20.0), Trace("noise-only", np.zeros(N_SAMPLES), 20.0)]
    with pytest.raises(TraceProcessingError, match="^noise-only: "):
        build_attribute_array(traces)


def test_empty_input():
    with pytest.raises(ValueError):
        build_attribute_array([])
