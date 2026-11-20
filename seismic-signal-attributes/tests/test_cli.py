import json
import subprocess
import sys

import numpy as np
import obspy

from seismic_signal_attributes.cli import main
from seismic_signal_attributes.pipeline import ATTRIBUTE_NAMES


def test_build_end_to_end(mseed_files, tmp_path):
    out = tmp_path / "out"
    assert main(["build", *map(str, mseed_files), "--output-dir", str(out), "--plot"]) == 0

    array = np.load(out / "attributes.npy")
    assert array.shape == (3, 12, 6000)
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["attribute_names"] == list(ATTRIBUTE_NAMES)
    first = meta["traces"][0]
    assert first["trace_id"] == "XX.R0.00.BHZ"
    assert abs(first["onset_sample"] - 3000) <= 5
    assert obspy.UTCDateTime(first["onset_utc"]) == obspy.UTCDateTime(2020, 1, 1) + first["onset_time_s"]
    assert sorted(p.name for p in (out / "plots").iterdir()) == [
        f"{i:03d}_XX.R{i}.00.BHZ.png" for i in range(3)
    ]


def test_bad_input_returns_error_code(tmp_path, caplog):
    assert main(["build", str(tmp_path / "missing"), "--output-dir", str(tmp_path / "out")]) == 2
    assert "no such file" in caplog.text


def test_console_script_help():
    result = subprocess.run(
        [sys.executable, "-m", "seismic_signal_attributes.cli", "build", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--sampling-rate" in result.stdout
