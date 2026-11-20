"""Kurtosis, gain and correlation attributes of seismic receiver traces.

Production version of the Colab notebook
``Seismic_Signal_Derivatives-Kurtosis_Auto_and_XcorrelationsGainFuncsCepstrum and array build.ipynb``.
"""

from seismic_signal_attributes.cepstrum import real_cepstrum
from seismic_signal_attributes.characteristic import (
    characteristic_functions,
    envelope,
    kurtosis_cf,
    spline_scale,
)
from seismic_signal_attributes.correlation import (
    aligned_correlation,
    aligned_correlation_sum,
    impulse,
    impulse_correlation,
)
from seismic_signal_attributes.gain import amplitude_recovery, amplitude_trend, preset_gain
from seismic_signal_attributes.io import Trace, load_trace, load_traces
from seismic_signal_attributes.onset import (
    OnsetNotFoundError,
    detect_onset,
    detect_subsequent_onsets,
)
from seismic_signal_attributes.pipeline import (
    ATTRIBUTE_NAMES,
    TraceProcessingError,
    build_attribute_array,
)

__version__ = "1.0.0"

__all__ = [
    "ATTRIBUTE_NAMES",
    "OnsetNotFoundError",
    "Trace",
    "TraceProcessingError",
    "aligned_correlation",
    "aligned_correlation_sum",
    "amplitude_recovery",
    "amplitude_trend",
    "build_attribute_array",
    "characteristic_functions",
    "detect_onset",
    "detect_subsequent_onsets",
    "envelope",
    "impulse",
    "impulse_correlation",
    "kurtosis_cf",
    "load_trace",
    "load_traces",
    "preset_gain",
    "real_cepstrum",
    "spline_scale",
]
