"""Gain functions applied relative to the signal onset."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit


def _check_onset(n: int, onset: int) -> None:
    if not 0 <= onset < n:
        raise ValueError(f"onset {onset} is outside the trace (0..{n - 1})")


def preset_gain(
    signal: np.ndarray, onset: int, hold: int = 200, time_constant: float = 200.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed-shape gain: ``1.01**(i - onset)`` before the onset, 1 for ``hold`` samples,
    then ``2 - exp((hold - (i - onset)) / time_constant)`` rising towards 2.

    Returns ``(signal * gain, gain)``.
    """
    n = len(signal)
    _check_onset(n, onset)
    position = np.arange(n) - onset
    gain = np.empty(n)
    gain[:onset] = np.power(1.01, position[:onset])
    gain[onset : onset + hold] = 1.0
    gain[onset + hold :] = 2 - np.exp((hold - position[onset + hold :]) / time_constant)
    return signal * gain, gain


def amplitude_trend(signal: np.ndarray, onset: int, step: int = 20) -> np.ndarray:
    """Peak-amplitude trend after the onset.

    The largest |signal| in each ``step``-sample block from the onset onwards is
    linearly interpolated over every post-onset sample. Samples before the onset are 1.
    """
    n = len(signal)
    _check_onset(n, onset)
    inverted = 1 - np.abs(signal)
    starts = np.arange(onset, n, step)
    knots = np.array([s + np.argmin(inverted[s : s + step]) for s in starts])
    trend = np.zeros(n)
    trend[onset:] = np.interp(np.arange(onset, n), knots, inverted[knots])
    return 1 - trend


def fit_gain(trend: np.ndarray, onset: int, maxfev: int = 9000) -> Tuple[float, float, float]:
    """Least-squares fit of ``a * exp(-b * (i - onset)) + c`` to ``trend[onset:]``."""
    _check_onset(len(trend), onset)

    def model(x, a, b, c):
        return a * np.exp(-b * (x - onset)) + c

    x = np.arange(onset, len(trend), dtype=np.float64)
    with np.errstate(over="ignore"):
        try:
            (a, b, c), _ = curve_fit(model, x, trend[onset:], maxfev=maxfev)
        except RuntimeError as exc:
            raise ValueError(f"gain curve fit did not converge: {exc}") from exc
    return float(a), float(b), float(c)


def gain_curve(params: Tuple[float, float, float], n: int, onset: int) -> np.ndarray:
    """The fitted curve from the onset onwards, scaled to a maximum of 1; 1 before the onset."""
    _check_onset(n, onset)
    a, b, c = params
    gain = np.ones(n)
    gain[onset:] = a * np.exp(-b * (np.arange(onset, n) - onset)) + c
    gain /= gain.max()
    gain[:onset] = 1.0
    if not np.all(np.isfinite(gain)) or np.any(gain <= 0):
        raise ValueError(f"fitted gain curve (a={a:.4g}, b={b:.4g}, c={c:.4g}) is not positive")
    return gain


def amplitude_recovery(
    signal: np.ndarray, onset: int, step: int = 20
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """SigSplineScaleAmpRec: the signal divided by a fitted exponential decay curve.

    Returns ``(recovered_signal, gain, (a, b, c))``.
    """
    params = fit_gain(amplitude_trend(signal, onset, step), onset)
    gain = gain_curve(params, len(signal), onset)
    return signal / gain, gain, params
