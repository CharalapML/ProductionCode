"""Reading waveform files with ObsPy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import obspy

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass
class Trace:
    """One receiver trace: raw samples plus the metadata the pipeline reports."""

    name: str
    data: np.ndarray
    sampling_rate: float
    trace_id: str = ""
    starttime: Optional[obspy.UTCDateTime] = None


def load_trace(path: PathLike) -> Trace:
    """Read the first trace in any ObsPy-supported file (miniSEED, SAC, ...)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    stream = obspy.read(str(path))
    if len(stream) > 1:
        log.warning("%s contains %d traces; using the first", path.name, len(stream))
    tr = stream[0]
    return Trace(
        name=path.name,
        data=np.asarray(tr.data, dtype=np.float64),
        sampling_rate=float(tr.stats.sampling_rate),
        trace_id=tr.id,
        starttime=tr.stats.starttime,
    )


def load_traces(paths: Sequence[PathLike], sampling_rate: Optional[float] = 20.0) -> List[Trace]:
    """Read every file; drop (with a warning) traces whose sampling rate differs from
    ``sampling_rate``. Pass ``None`` to keep all."""
    traces = []
    for path in paths:
        trace = load_trace(path)
        if sampling_rate is not None and trace.sampling_rate != sampling_rate:
            log.warning(
                "skipping %s: sampling rate %g Hz, expected %g Hz",
                trace.name, trace.sampling_rate, sampling_rate,
            )
            continue
        traces.append(trace)
    if not traces:
        raise ValueError("no traces left to process")
    return traces
