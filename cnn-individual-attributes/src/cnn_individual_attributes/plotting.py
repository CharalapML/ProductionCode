"""Per-receiver diagnostic plots (requires the ``plot`` extra: matplotlib)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from cnn_individual_attributes.postprocess import smooth_prediction, threshold


def plot_event(
    prediction: np.ndarray,
    attribute: np.ndarray,
    out_dir: Union[str, Path],
    target: Optional[np.ndarray] = None,
    window: int = 20,
    threshold_factor: float = 1.5,
    display_scale: float = 10.0,
    prefix: str = "event",
) -> List[Path]:
    """Save one PNG per receiver for a single event.

    ``prediction``, ``attribute`` and ``target`` are ``(num_receivers, trace_length)``.
    As in the notebook, the attribute and target are divided by ``display_scale``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    smoothed = smooth_prediction(prediction, window)
    levels = threshold(prediction, threshold_factor)
    paths = []
    for r in range(prediction.shape[0]):
        fig, ax = plt.subplots(figsize=(35, 5))
        ax.set_title(f"{prefix} receiver {r}", size=20)
        ax.plot(np.abs(prediction[r]), label="|prediction|", color="red")
        if target is not None:
            ax.plot(target[r] / display_scale, label="target", color="black")
        ax.plot(attribute[r] / display_scale, label="input attribute", color="blue")
        ax.plot(smoothed[r], label=f"moving mean (window={window})", color="grey")
        ax.axhline(levels[r], linestyle="--", color="red", label=f"{threshold_factor} x mean")
        ax.legend(loc="lower right")
        path = out_dir / f"{prefix}_receiver{r:02d}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
