"""Build the ``(traces, attributes, samples)`` array for a set of receiver traces."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from seismic_signal_attributes.characteristic import characteristic_functions, spline_scale
from seismic_signal_attributes.correlation import (
    aligned_correlation,
    aligned_correlation_sum,
    impulse,
    impulse_correlation,
)
from seismic_signal_attributes.gain import amplitude_recovery, preset_gain
from seismic_signal_attributes.io import Trace
from seismic_signal_attributes.onset import detect_onset, detect_subsequent_onsets

log = logging.getLogger(__name__)

ATTRIBUTE_NAMES: Tuple[str, ...] = (
    "SigSplineScale",            # 0  spline-detrended signal scaled to max |x| = 1
    "F1cf",                      # 1  kurtosis characteristic function
    "F4cf",                      # 2
    "F4cfgrad",                  # 3
    "F4cfgrad_gtmean",           # 4
    "env_pos",                   # 5  approximate positive envelope
    "SigSplineScalePreSetGain",  # 6  signal × fixed-shape gain
    "XcorrShiftAdded",           # 7  sum of onset-aligned cross-correlations with the other traces
    "AllImpACorr",               # 8  impulse autocorrelation
    "AXcorrShiftAdded",          # 9  as 7, plus the trace's own autocorrelation
    "SigSplineScaleAmpRec",      # 10 signal ÷ fitted exponential decay
    "AllImpXcorrAdded",          # 11 sum of impulse cross-correlations with the other traces
)

F1_LEAD_WARNING_S = 30.0


class TraceProcessingError(ValueError):
    """A trace could not be processed; the message starts with the trace name."""


def build_attribute_array(traces: Sequence[Trace]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Compute all :data:`ATTRIBUTE_NAMES` for each trace.

    Returns ``(array, metadata)``. ``array`` has shape
    ``(len(traces), len(ATTRIBUTE_NAMES), longest_trace)``; shorter traces are
    zero-padded at the end. ``metadata`` has one dict per trace with its onset,
    later F4cf onsets and fitted gain parameters.
    """
    if not traces:
        raise ValueError("no traces given")

    scaled, features, onsets, later = [], [], [], []
    for trace in traces:
        try:
            signal = spline_scale(trace.data)
            cf = characteristic_functions(signal)
            args = (cf["F1cf"], cf["F4cf"], cf["F4cfgrad_gtmean"], cf["threshold"])
            onset = detect_onset(*args)
            subsequent = detect_subsequent_onsets(*args)
        except ValueError as exc:
            raise TraceProcessingError(f"{trace.name}: {exc}") from exc
        lead = (np.argmax(np.abs(signal)) - np.argmax(cf["F1cf"])) / trace.sampling_rate
        if lead > F1_LEAD_WARNING_S:
            log.warning(
                "%s: F1cf peak is %.1f s before the largest amplitude; review this trace",
                trace.name, lead,
            )
        log.info("%s: onset at sample %d", trace.name, onset)
        scaled.append(signal)
        features.append(cf)
        onsets.append(onset)
        later.append(subsequent)

    length = max(len(s) for s in scaled)
    array = np.zeros((len(traces), len(ATTRIBUTE_NAMES), length))
    metadata = []
    for i, trace in enumerate(traces):
        signal, cf, onset = scaled[i], features[i], onsets[i]
        others = [scaled[k] for k in range(len(traces)) if k != i]
        try:
            xcorr_sum = aligned_correlation_sum(signal, others, onset)
            amp_rec, _, gain_params = amplitude_recovery(signal, onset)
            attributes = {
                "SigSplineScale": signal,
                "F1cf": cf["F1cf"],
                "F4cf": cf["F4cf"],
                "F4cfgrad": cf["F4cfgrad"],
                "F4cfgrad_gtmean": cf["F4cfgrad_gtmean"],
                "env_pos": cf["env_pos"],
                "SigSplineScalePreSetGain": preset_gain(signal, onset)[0],
                "XcorrShiftAdded": xcorr_sum,
                "AllImpACorr": impulse_correlation(signal, onset, impulse(signal, onset)),
                "AXcorrShiftAdded": xcorr_sum + aligned_correlation(signal, signal, onset),
                "SigSplineScaleAmpRec": amp_rec,
                "AllImpXcorrAdded": sum(
                    (
                        impulse_correlation(signal, onset, impulse(scaled[k], onsets[k]))
                        for k in range(len(traces))
                        if k != i
                    ),
                    np.zeros(len(signal)),
                ),
            }
        except ValueError as exc:
            raise TraceProcessingError(f"{trace.name}: {exc}") from exc

        for a, name in enumerate(ATTRIBUTE_NAMES):
            array[i, a, : len(signal)] = attributes[name]

        delta = 1.0 / trace.sampling_rate
        metadata.append(
            {
                "file": trace.name,
                "trace_id": trace.trace_id,
                "sampling_rate": trace.sampling_rate,
                "npts": len(signal),
                "starttime": str(trace.starttime) if trace.starttime is not None else None,
                "onset_sample": onset,
                "onset_time_s": onset * delta,
                "onset_utc": str(trace.starttime + onset * delta) if trace.starttime is not None else None,
                "f4cfgrad_threshold": cf["threshold"],
                "subsequent_onset_samples": later[i].tolist(),
                "subsequent_onset_offsets_s": ((later[i] - onset) * delta).tolist(),
                "gain_fit": dict(zip("abc", gain_params)),
            }
        )
    return array, metadata
