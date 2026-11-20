"""Onset-aligned cross-/auto-correlations between receiver traces."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import correlate

IMPULSE_LENGTH = 60  # 3 s at 20 Hz


def _shift_left(x: np.ndarray, shift: int) -> np.ndarray:
    """``out[i] = x[i + shift]`` where that index exists, else 0 (no wrap-around)."""
    out = np.zeros_like(x)
    if shift >= 0:
        out[: len(x) - shift] = x[shift:]
    else:
        out[-shift:] = x[: len(x) + shift]
    return out


def aligned_correlation(target: np.ndarray, other: np.ndarray, onset: int) -> np.ndarray:
    """Full cross-correlation of ``target`` with ``other``, shifted so its largest
    |value| sits at sample ``onset``, truncated to ``len(target)``."""
    xc = correlate(target, other, mode="full")
    peak = int(np.argmax(xc)) if abs(xc.max()) > abs(xc.min()) else int(np.argmin(xc))
    return _shift_left(xc, peak - onset)[: len(target)]


def aligned_correlation_sum(target: np.ndarray, others: Sequence[np.ndarray], onset: int) -> np.ndarray:
    """Sum of :func:`aligned_correlation` of ``target`` with each of ``others``."""
    total = np.zeros(len(target))
    for other in others:
        total += aligned_correlation(target, other, onset)
    return total


def impulse(signal: np.ndarray, onset: int, length: int = IMPULSE_LENGTH) -> np.ndarray:
    """The ``length`` samples starting at ``onset``."""
    if onset < 0 or onset + length > len(signal):
        raise ValueError(
            f"a {length}-sample impulse from onset {onset} does not fit in {len(signal)} samples"
        )
    return signal[onset : onset + length]


def impulse_correlation(signal: np.ndarray, onset: int, pulse: np.ndarray) -> np.ndarray:
    """Slide ``pulse`` along ``signal`` (zeroed before ``onset``).

    ``out[j] = sum(signal[j:j+L] * pulse)`` for ``j < len(signal) - L``; the last
    ``L = len(pulse)`` samples are 0.
    """
    n, length = len(signal), len(pulse)
    zeroed = np.array(signal, dtype=np.float64)
    zeroed[:onset] = 0
    out = np.zeros(n)
    out[: n - length] = np.correlate(zeroed, pulse, mode="valid")[: n - length]
    return out
