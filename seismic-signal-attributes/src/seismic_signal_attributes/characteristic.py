"""Kurtosis-based characteristic functions F1cf–F4cf and their derived attributes.

Based on Baillard et al. (2014), "An Automatic Kurtosis-Based P- and S-Phase Picker
Designed for Local Seismic Networks", BSSA 104(1), 394–409. The scale factors
(``/4`` on F1cf, ``×5`` on F4cf, ``×8`` on the gradient threshold, ``×1.5`` on the
envelope) are kept exactly as in the original notebook so the arrays match the data
the downstream models were built on.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from obspy.signal.detrend import spline

KURTOSIS_WINDOW = 100
ENVELOPE_WINDOW = 8


def spline_scale(data: np.ndarray, order: int = 2, dspline: int = 1000) -> np.ndarray:
    """Remove a spline trend (``obspy.signal.detrend.spline``) and scale to max |x| = 1."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError(f"expected a 1-D trace, got shape {data.shape}")
    if len(data) < 2 * dspline:
        raise ValueError(f"trace has {len(data)} samples; at least {2 * dspline} are needed")
    detrended = spline(data.copy(), order=order, dspline=dspline, plot=False)
    peak = np.abs(detrended).max()
    if not np.isfinite(peak) or peak == 0:
        raise ValueError("trace is flat or non-finite after detrending")
    return detrended / peak


def _trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Mean of ``x[i-window+1 : i+1]`` at each ``i >= window-1``; zero before that."""
    out = np.zeros(len(x))
    out[window - 1 :] = sliding_window_view(x, window).mean(axis=1)
    return out


def kurtosis_cf(signal: np.ndarray, window: int = KURTOSIS_WINDOW) -> np.ndarray:
    """F1cf: sliding-window kurtosis (divided by 4), zero for the first ``2*(window-1)`` samples."""
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < 2 * window:
        raise ValueError(f"signal has {len(signal)} samples; at least {2 * window} are needed")
    start = window - 1
    deviation = np.zeros(len(signal))
    deviation[start:] = signal[start:] - _trailing_mean(signal, window)[start:]
    e4mean = _trailing_mean(deviation**4, window)
    e2mean = _trailing_mean(deviation**2, window)
    f1 = np.zeros(len(signal))
    valid = e2mean > 0
    f1[valid] = e4mean[valid] / e2mean[valid] ** 2 / 4
    f1[: 2 * start] = 0
    return f1


def f2_cf(f1: np.ndarray) -> np.ndarray:
    """F2cf: cumulative sum of the non-negative increments of F1cf."""
    return np.concatenate(([0.0], np.cumsum(np.maximum(np.diff(f1), 0))))


def f3_cf(f2: np.ndarray) -> np.ndarray:
    """F3cf: F2cf minus the straight line joining its first and last values."""
    slope = (f2[-1] - f2[0]) / (len(f2) - 1)
    return f2 - slope * np.arange(len(f2))


def f4_cf(f3: np.ndarray) -> np.ndarray:
    """F4cf: ``5 × (F3[m+1] − F3[m])`` where F3 is rising, else 0 (last sample is 0)."""
    f4 = np.zeros(len(f3))
    rising = f3[:-1] < f3[1:]
    f4[:-1] = np.where(rising, (f3[:-1] - f3[1:]) * -5, 0.0)
    return f4


def f4cf_gradient(f4: np.ndarray) -> np.ndarray:
    """F4cfgrad: positive part of ``np.gradient(F4cf)``."""
    return np.maximum(np.gradient(f4), 0.0)


def gradient_threshold(f4cfgrad: np.ndarray, factor: float = 8.0) -> float:
    """``factor`` × mean of the strictly positive F4cfgrad values (0 if there are none).

    Called ``F4cfgradmeanx2`` in the notebook, although the factor there is 8.
    """
    positive = f4cfgrad[f4cfgrad > 0]
    return float(positive.mean() * factor) if positive.size else 0.0


def above_threshold(f4cfgrad: np.ndarray, threshold: float) -> np.ndarray:
    """F4cfgrad_gtmean: F4cfgrad where it exceeds ``threshold``, else 0."""
    return np.where(f4cfgrad > threshold, f4cfgrad, 0.0)


def envelope(signal: np.ndarray, window: int = ENVELOPE_WINDOW, scale: float = 1.5) -> np.ndarray:
    """env_pos: trailing ``window``-sample mean of |signal| × ``scale``; zero before the first full window."""
    return _trailing_mean(np.abs(signal), window) * scale


def characteristic_functions(
    signal: np.ndarray,
    window: int = KURTOSIS_WINDOW,
    threshold_factor: float = 8.0,
) -> Dict[str, np.ndarray]:
    """All single-trace kurtosis attributes for a spline-scaled signal.

    Returns a dict with ``F1cf``, ``F2cf``, ``F3cf``, ``F4cf``, ``F4cfgrad``,
    ``F4cfgrad_gtmean``, ``env_pos`` (arrays) and ``threshold`` (float).
    """
    f1 = kurtosis_cf(signal, window)
    f2 = f2_cf(f1)
    f3 = f3_cf(f2)
    f4 = f4_cf(f3)
    grad = f4cf_gradient(f4)
    threshold = gradient_threshold(grad, threshold_factor)
    return {
        "F1cf": f1,
        "F2cf": f2,
        "F3cf": f3,
        "F4cf": f4,
        "F4cfgrad": grad,
        "F4cfgrad_gtmean": above_threshold(grad, threshold),
        "env_pos": envelope(signal),
        "threshold": threshold,
    }
