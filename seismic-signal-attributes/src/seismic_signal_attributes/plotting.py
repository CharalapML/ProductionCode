"""Diagnostic plot of every attribute of one trace around its onset (needs matplotlib)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np


def plot_trace(
    attributes: np.ndarray,
    names: Sequence[str],
    onset: int,
    path: Union[str, Path],
    title: str = "",
    before: int = 500,
    after: int = 2000,
) -> Path:
    """Save one stacked panel per attribute, from ``onset - before`` to ``onset + after``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start, stop = max(onset - before, 0), min(onset + after, attributes.shape[1])
    x = np.arange(start, stop)
    fig, axes = plt.subplots(len(names), 1, figsize=(14, 1.6 * len(names)), sharex=True)
    for ax, row, name in zip(np.atleast_1d(axes), attributes, names):
        ax.plot(x, row[start:stop], linewidth=0.8)
        ax.axvline(onset, color="red", linewidth=0.8)
        ax.set_ylabel(name, rotation=0, ha="right", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("sample")
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
