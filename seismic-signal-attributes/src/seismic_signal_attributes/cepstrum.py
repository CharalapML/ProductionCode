"""Real cepstrum of a trace segment (exploratory; not part of the attribute array)."""

from __future__ import annotations

import numpy as np


def real_cepstrum(segment: np.ndarray, hamming: bool = True) -> np.ndarray:
    """``irfft(log|rfft(segment × window)|)``.

    Index ``k`` of the result is quefrency ``k / sampling_rate`` seconds. Zero spectral
    magnitudes are floored at the smallest positive float so the log stays finite.
    """
    segment = np.asarray(segment, dtype=np.float64)
    if segment.ndim != 1 or len(segment) < 2:
        raise ValueError("segment must be a 1-D array of at least 2 samples")
    if hamming:
        segment = segment * np.hamming(len(segment))
    magnitude = np.abs(np.fft.rfft(segment))
    log_magnitude = np.log(np.maximum(magnitude, np.finfo(np.float64).tiny))
    return np.fft.irfft(log_magnitude, n=len(segment))
