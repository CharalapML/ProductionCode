import numpy as np
import pytest

from cnn_individual_attributes.data import (
    attribute_index,
    load_dataset,
    prepare_inputs,
    prepare_targets,
    select_attribute,
    train_test_split,
)


def test_attribute_index_by_name_index_and_digit_string():
    assert attribute_index("F4cfgrad_gtmean") == 4
    assert attribute_index("signal") == 0
    assert attribute_index("f1CF") == 1
    assert attribute_index(2) == 2
    assert attribute_index("3") == 3


@pytest.mark.parametrize("bad", ["nope", -1])
def test_attribute_index_rejects_unknown(bad):
    with pytest.raises(ValueError):
        attribute_index(bad)


def test_load_dataset_replaces_target_nans(synthetic_files):
    inputs, targets = load_dataset(*synthetic_files)
    assert inputs.shape == (4, 3, 5, 80)
    assert targets.shape == (4, 3, 1, 80)
    assert not np.isnan(targets).any()
    assert targets[0, 0, 0, :5].tolist() == [0.0] * 5


def test_load_dataset_inputs_only(synthetic_files):
    inputs, targets = load_dataset(synthetic_files[0])
    assert targets is None and inputs.ndim == 4


def test_load_dataset_rejects_mismatched_shapes(tmp_path, synthetic_arrays):
    inputs, targets = synthetic_arrays
    np.save(tmp_path / "in.npy", inputs)
    np.save(tmp_path / "out.npy", targets[:, :, :, :40])
    with pytest.raises(ValueError, match="disagree"):
        load_dataset(tmp_path / "in.npy", tmp_path / "out.npy")


def test_inputs_with_nan_rejected(synthetic_arrays):
    inputs = synthetic_arrays[0].copy()
    inputs[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        prepare_inputs(inputs)


def test_select_and_prepare_shapes(synthetic_arrays):
    inputs, targets = synthetic_arrays
    sel = select_attribute(inputs, "F4cf")
    assert sel.shape == (4, 3, 1, 80)
    np.testing.assert_array_equal(sel[:, :, 0], inputs[:, :, 2])

    x = prepare_inputs(inputs)
    assert x.shape == (4, 3, 1, 80, 1) and x.dtype == np.float32
    np.testing.assert_array_equal(x[..., 0, :, 0], inputs[:, :, 4])

    y = prepare_targets(targets)
    assert y.shape == (4, 3, 1, 80, 1) and not np.isnan(y).any()


def test_select_attribute_out_of_range(synthetic_arrays):
    with pytest.raises(ValueError, match="out of range"):
        select_attribute(synthetic_arrays[0][:, :, :2], 4)


def test_train_test_split_sizes_and_reproducible():
    x = np.arange(4)
    y = x * 10
    xtr, xte, ytr, yte, tr, te = train_test_split(x, y, test_size=0.25, seed=1)
    assert len(xte) == 1 and len(xtr) == 3  # same as the notebook's 3/1 split
    assert sorted(np.concatenate([tr, te]).tolist()) == [0, 1, 2, 3]
    np.testing.assert_array_equal(ytr, xtr * 10)
    again = train_test_split(x, y, test_size=0.25, seed=1)
    np.testing.assert_array_equal(again[5], te)


@pytest.mark.parametrize("test_size", [0.0, 1.0, 0.9])
def test_train_test_split_rejects_bad_size(test_size):
    with pytest.raises(ValueError):
        train_test_split(np.arange(4), np.arange(4), test_size=test_size)
