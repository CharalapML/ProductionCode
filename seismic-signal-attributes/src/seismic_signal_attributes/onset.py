"""Signal onset picking from the F1cf peak and the F4cf / F4cfgrad attributes."""

from __future__ import annotations

import numpy as np


class OnsetNotFoundError(ValueError):
    """No sample near the F1cf peak satisfies the onset criteria."""


def detect_onset(
    f1cf: np.ndarray,
    f4cf: np.ndarray,
    f4cfgrad_gtmean: np.ndarray,
    threshold: float,
    search: int = 7,
    lookback: int = 7,
) -> int:
    """Onset sample of the main arrival.

    Starting at the F1cf maximum and stepping back ``search`` samples, every sample
    ``x`` with ``F4cfgrad_gtmean[x] > threshold`` proposes the nearest ``x - j``
    (``1 <= j <= lookback``) where ``F4cf`` is exactly 0. The onset is the latest
    proposal.
    """
    peak = int(np.argmax(f1cf))
    candidates = []
    for x in range(peak, max(peak - search, -1), -1):
        if not f4cfgrad_gtmean[x] > threshold:
            continue
        for j in range(1, lookback + 1):
            if x - j < 0:
                break
            if f4cf[x - j] == 0:
                candidates.append(x - j)
                break
    if not candidates:
        raise OnsetNotFoundError(
            f"no onset found within {search} samples before the F1cf peak at sample {peak}"
        )
    return max(candidates)


def detect_subsequent_onsets(
    f1cf: np.ndarray,
    f4cf: np.ndarray,
    f4cfgrad_gtmean: np.ndarray,
    threshold: float,
    lookback: int = 5,
) -> np.ndarray:
    """Sorted sample indices of later F4cf events, searching forward from the F1cf peak.

    For each ``x`` with ``F4cfgrad_gtmean[x] > threshold``: propose ``x - 1`` if
    ``F4cf[x-1] == 0``, otherwise the nearest ``x - j`` (``2 <= j <= lookback``)
    where both ``F4cf`` and ``F4cfgrad_gtmean`` are 0.
    """
    peak = int(np.argmax(f1cf))
    found = set()
    for x in range(max(peak, 1), len(f1cf)):
        if not f4cfgrad_gtmean[x] > threshold:
            continue
        if f4cf[x - 1] == 0:
            found.add(x - 1)
            continue
        for j in range(2, lookback + 1):
            if x - j < 0:
                break
            if f4cfgrad_gtmean[x - j] == 0 and f4cf[x - j] == 0:
                found.add(x - j)
                break
    return np.array(sorted(found), dtype=int)
