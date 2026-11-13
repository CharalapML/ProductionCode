"""Post-processing of predictions, as in the last cells of the notebook.

For each receiver trace: take the absolute prediction, smooth it with a
trailing moving mean, roll it back by half a window to centre it, and compare
against a threshold of ``factor * mean(|prediction|)``.
"""

from __future__ import annotations

import numpy as np


def moving_mean(a: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving mean along the last axis.

    ``out[..., i] = mean(a[..., i-window+1 : i+1])``; the first ``window - 1``
    samples are NaN (same behaviour as ``bottleneck.move_mean(a, window)``).
    """
    a = np.asarray(a, dtype=np.float64)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window > a.shape[-1]:
        raise ValueError(f"window {window} is longer than the trace ({a.shape[-1]})")
    csum = np.cumsum(a, axis=-1)
    out = np.full(a.shape, np.nan)
    out[..., window - 1] = csum[..., window - 1]
    out[..., window:] = csum[..., window:] - csum[..., :-window]
    out[..., window - 1 :] /= window
    return out


def smooth_prediction(prediction: np.ndarray, window: int = 20) -> np.ndarray:
    """Absolute value, moving mean, then a roll of ``round(-window / 2)`` samples."""
    smoothed = moving_mean(np.abs(prediction), window)
    return np.roll(smoothed, round(-window / 2), axis=-1)


def threshold(prediction: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """``factor * mean(|prediction|)`` along the last axis (one value per trace)."""
    return factor * np.mean(np.abs(prediction), axis=-1)
