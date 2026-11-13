import numpy as np
import pytest

# Small shapes keep the tests fast; trace_length must be a multiple of 40.
N_EVENTS, N_RECEIVERS, N_ATTRIBUTES, TRACE_LENGTH = 4, 3, 5, 80


@pytest.fixture
def synthetic_arrays():
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(N_EVENTS, N_RECEIVERS, N_ATTRIBUTES, TRACE_LENGTH)).astype(np.float32)
    targets = np.zeros((N_EVENTS, N_RECEIVERS, 1, TRACE_LENGTH), dtype=np.float32)
    targets[:, :, 0, 30:40] = 1.0
    targets[0, 0, 0, :5] = np.nan  # the real targets contain NaNs
    return inputs, targets


@pytest.fixture
def synthetic_files(tmp_path, synthetic_arrays):
    inputs, targets = synthetic_arrays
    in_path, out_path = tmp_path / "in.npy", tmp_path / "out.npy"
    np.save(in_path, inputs)
    np.save(out_path, targets)
    return in_path, out_path
