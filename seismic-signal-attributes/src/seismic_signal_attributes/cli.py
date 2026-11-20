"""Command-line interface: ``seismic-signal-attributes build``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from seismic_signal_attributes.io import load_traces
from seismic_signal_attributes.pipeline import ATTRIBUTE_NAMES, build_attribute_array

log = logging.getLogger("seismic_signal_attributes")


def _build(args: argparse.Namespace) -> int:
    rate = None if args.sampling_rate == 0 else args.sampling_rate
    traces = load_traces(args.traces, sampling_rate=rate)
    array, metadata = build_attribute_array(traces)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "attributes.npy", array)
    (out / "metadata.json").write_text(
        json.dumps({"attribute_names": list(ATTRIBUTE_NAMES), "traces": metadata}, indent=2)
    )
    log.info("saved attributes %s and metadata.json to %s", array.shape, out)

    if args.plot:
        from seismic_signal_attributes.plotting import plot_trace

        for i, meta in enumerate(metadata):
            plot_trace(
                array[i, :, : meta["npts"]],
                ATTRIBUTE_NAMES,
                meta["onset_sample"],
                out / "plots" / f"{i:03d}_{meta['file']}.png",
                title=meta["file"],
            )
        log.info("saved plots to %s", out / "plots")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seismic-signal-attributes", description=__doc__)
    parser.add_argument("-v", "--verbose", action="count", default=0, help="more logging")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="compute the attribute array for a set of receiver traces")
    b.add_argument("traces", nargs="+", help="waveform files readable by ObsPy (miniSEED, SAC, ...)")
    b.add_argument("--output-dir", required=True, help="directory for attributes.npy and metadata.json")
    b.add_argument(
        "--sampling-rate", type=float, default=20.0,
        help="skip traces with any other rate, in Hz (default: 20; 0 keeps all)",
    )
    b.add_argument("--plot", action="store_true", help="save one PNG per trace (needs matplotlib)")
    b.set_defaults(func=_build)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose > 1 else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
