import numpy as np
import obspy
import pytest

from seismic_signal_attributes.io import Trace

SAMPLING_RATE = 20.0
N_SAMPLES = 6000
ARRIVALS = (3000, 3037, 3074)  # one per receiver


def synthetic_signal(seed: int, n: int = N_SAMPLES, arrival: int = ARRIVALS[0]) -> np.ndarray:
    """Noise + slow trend + a decaying wavelet at ``arrival`` and a weaker,
    reversed-polarity 'depth phase' 200 samples later."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = rng.normal(0, 0.02, n) + 1e-8 * (t - n / 2) ** 2
    for start, amplitude in ((arrival, 1.0), (arrival + 200, -0.5)):
        tt = t - start
        after = tt >= 0
        x[after] += amplitude * np.exp(-tt[after] / 60) * np.sin(2 * np.pi * tt[after] / 12)
    return x


@pytest.fixture
def traces():
    return [
        Trace(f"XX.R{i}.00.BHZ", synthetic_signal(i, arrival=a), SAMPLING_RATE, f"XX.R{i}.00.BHZ",
              obspy.UTCDateTime(2020, 1, 1))
        for i, a in enumerate(ARRIVALS)
    ]


@pytest.fixture
def mseed_files(tmp_path):
    paths = []
    for i, arrival in enumerate(ARRIVALS):
        tr = obspy.Trace(
            data=(synthetic_signal(i, arrival=arrival) * 1e4).astype(np.int32),
            header={"network": "XX", "station": f"R{i}", "location": "00", "channel": "BHZ",
                    "sampling_rate": SAMPLING_RATE, "starttime": obspy.UTCDateTime(2020, 1, 1)},
        )
        path = tmp_path / f"XX.R{i}.00.BHZ"
        tr.write(str(path), format="MSEED")
        paths.append(path)
    return paths
