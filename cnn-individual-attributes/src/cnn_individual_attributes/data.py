"""Loading, validating and shaping the 4D input/target arrays.

Array layout (as produced for the original notebook):

* inputs:  ``(num_events, num_receivers, num_attributes, trace_length)``
* targets: ``(num_events, num_receivers, 1, trace_length)``

The model consumes a single attribute with an extra trailing channel axis:
``(num_events, num_receivers, 1, trace_length, 1)``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple, Union

import numpy as np

# Attribute order along axis 2 of the input array.
ATTRIBUTE_NAMES = (
    "signal",  # original splined, scaled signal
    "F1cf",
    "F4cf",
    "F4cfgrad",
    "F4cfgrad_gtmean",
)
DEFAULT_ATTRIBUTE = "F4cfgrad_gtmean"

AttributeRef = Union[int, str]


def attribute_index(attribute: AttributeRef) -> int:
    """Return the axis-2 index for an attribute given by name or index."""
    if isinstance(attribute, str):
        if attribute.isdigit():
            attribute = int(attribute)
        else:
            lookup = {name.lower(): i for i, name in enumerate(ATTRIBUTE_NAMES)}
            try:
                return lookup[attribute.lower()]
            except KeyError:
                raise ValueError(
                    f"Unknown attribute {attribute!r}; expected one of {ATTRIBUTE_NAMES} "
                    "or an integer index"
                ) from None
    if attribute < 0:
        raise ValueError(f"Attribute index must be non-negative, got {attribute}")
    return attribute


def load_dataset(
    inputs_path: Union[str, Path], targets_path: Union[str, Path, None] = None
) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """Load the input array and (optionally) the target array from ``.npy`` files.

    NaNs in the targets are replaced by 0, as in the original notebook. Inputs
    must be finite. Shapes are validated against each other.
    """
    inputs = np.load(inputs_path, allow_pickle=False)
    validate_inputs(inputs)
    if targets_path is None:
        return inputs, None
    targets = np.load(targets_path, allow_pickle=False)
    targets = clean_targets(targets)
    validate_pair(inputs, targets)
    return inputs, targets


def validate_inputs(inputs: np.ndarray) -> None:
    if inputs.ndim != 4:
        raise ValueError(
            "Inputs must be 4D (events, receivers, attributes, trace_length), "
            f"got shape {inputs.shape}"
        )
    if not np.all(np.isfinite(inputs)):
        raise ValueError("Inputs contain NaN or infinite values")


def clean_targets(targets: np.ndarray) -> np.ndarray:
    """Replace NaNs in the targets with 0."""
    if targets.ndim != 4 or targets.shape[2] != 1:
        raise ValueError(
            "Targets must be 4D (events, receivers, 1, trace_length), "
            f"got shape {targets.shape}"
        )
    return np.nan_to_num(targets, copy=True, nan=0.0)


def validate_pair(inputs: np.ndarray, targets: np.ndarray) -> None:
    n_in, r_in, _, t_in = inputs.shape
    n_out, r_out, _, t_out = targets.shape
    if (n_in, r_in, t_in) != (n_out, r_out, t_out):
        raise ValueError(
            f"Inputs {inputs.shape} and targets {targets.shape} disagree on "
            "events, receivers or trace_length"
        )


def select_attribute(inputs: np.ndarray, attribute: AttributeRef = DEFAULT_ATTRIBUTE) -> np.ndarray:
    """Slice one attribute, keeping axis 2: ``(N, R, A, T) -> (N, R, 1, T)``."""
    idx = attribute_index(attribute)
    if idx >= inputs.shape[2]:
        raise ValueError(
            f"Attribute index {idx} out of range for inputs with {inputs.shape[2]} attributes"
        )
    return inputs[:, :, idx : idx + 1, :]


def prepare_inputs(inputs: np.ndarray, attribute: AttributeRef = DEFAULT_ATTRIBUTE) -> np.ndarray:
    """Select an attribute and add the CNN channel axis: ``(N, R, 1, T, 1)`` float32."""
    validate_inputs(inputs)
    return np.expand_dims(select_attribute(inputs, attribute), axis=-1).astype(np.float32)


def prepare_targets(targets: np.ndarray) -> np.ndarray:
    """Clean targets and add the channel axis to match the model output: ``(N, R, 1, T, 1)``."""
    return np.expand_dims(clean_targets(targets), axis=-1).astype(np.float32)


def train_test_split(
    x: np.ndarray, y: np.ndarray, test_size: float = 0.25, seed: Union[int, None] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle events and split into train/test sets.

    The test set has ``ceil(test_size * N)`` events (the same rule as
    scikit-learn). Returns ``x_train, x_test, y_train, y_test, train_idx, test_idx``.
    """
    n = len(x)
    if len(y) != n:
        raise ValueError(f"x has {n} events but y has {len(y)}")
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    n_test = math.ceil(test_size * n)
    if n_test >= n:
        raise ValueError(f"test_size={test_size} leaves no training events out of {n}")
    perm = np.random.default_rng(seed).permutation(n)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx], train_idx, test_idx
