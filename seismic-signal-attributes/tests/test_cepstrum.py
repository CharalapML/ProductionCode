import numpy as np
import pytest

from seismic_signal_attributes.cepstrum import real_cepstrum


def test_echo_shows_up_at_its_delay():
    rng = np.random.default_rng(0)
    pulse = rng.normal(size=1000)
    delay = 37
    signal = pulse.copy()
    signal[delay:] += 0.6 * pulse[:-delay]
    cep = real_cepstrum(signal, hamming=False)
    assert len(cep) == 1000
    assert np.argmax(cep[10:500]) + 10 == delay


def test_zero_segment_is_finite():
    assert np.all(np.isfinite(real_cepstrum(np.zeros(64))))


def test_rejects_bad_segment():
    with pytest.raises(ValueError):
        real_cepstrum(np.zeros((2, 2)))
