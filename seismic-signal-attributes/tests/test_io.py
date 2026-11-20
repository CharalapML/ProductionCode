import logging

import obspy
import pytest

from seismic_signal_attributes.io import load_trace, load_traces


def test_load_trace(mseed_files):
    trace = load_trace(mseed_files[0])
    assert trace.name == "XX.R0.00.BHZ"
    assert trace.trace_id == "XX.R0.00.BHZ"
    assert trace.sampling_rate == 20.0
    assert trace.data.dtype.kind == "f" and len(trace.data) == 6000
    assert trace.starttime == obspy.UTCDateTime(2020, 1, 1)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_trace(tmp_path / "nope.mseed")


def test_other_sampling_rates_are_skipped(mseed_files, caplog):
    st = obspy.read(str(mseed_files[1]))
    st[0].stats.sampling_rate = 40.0
    st.write(str(mseed_files[1]), format="MSEED")
    with caplog.at_level(logging.WARNING):
        traces = load_traces(mseed_files)
    assert [t.name for t in traces] == ["XX.R0.00.BHZ", "XX.R2.00.BHZ"]
    assert "40 Hz" in caplog.text
    assert len(load_traces(mseed_files, sampling_rate=None)) == 3


def test_no_traces_left(mseed_files):
    with pytest.raises(ValueError, match="no traces"):
        load_traces(mseed_files, sampling_rate=100.0)


def test_multi_trace_file_uses_first(mseed_files, tmp_path, caplog):
    st = obspy.read(str(mseed_files[0])) + obspy.read(str(mseed_files[1]))
    path = tmp_path / "two.mseed"
    st.write(str(path), format="MSEED")
    with caplog.at_level(logging.WARNING):
        assert load_trace(path).trace_id == "XX.R0.00.BHZ"
    assert "2 traces" in caplog.text
