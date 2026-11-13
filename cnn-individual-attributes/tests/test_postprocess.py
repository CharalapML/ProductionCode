import numpy as np
import pytest

from cnn_individual_attributes.postprocess import moving_mean, smooth_prediction, threshold


def test_moving_mean_matches_bottleneck_example():
    # Example from the notebook: bn.move_mean([10,5,8,...], window=4)
    data = np.array([10, 5, 8, 9, 15, 22, 26, 11, 15, 16, 18, 7])
    out = moving_mean(data, 4)
    assert np.isnan(out[:3]).all()
    expected = [np.mean(data[i - 3 : i + 1]) for i in range(3, len(data))]
    np.testing.assert_allclose(out[3:], expected)


def test_moving_mean_works_on_last_axis_of_nd():
    a = np.arange(24, dtype=float).reshape(2, 3, 4)
    out = moving_mean(a, 2)
    np.testing.assert_allclose(out[..., 1:], (a[..., 1:] + a[..., :-1]) / 2)


@pytest.mark.parametrize("window", [0, 6])
def test_moving_mean_rejects_bad_window(window):
    with pytest.raises(ValueError):
        moving_mean(np.ones(5), window)


def test_smooth_prediction_is_abs_then_centred():
    pred = -np.ones((2, 100))
    pred[:, 50:60] = -5.0
    out = smooth_prediction(pred, window=20)
    expected = np.roll(moving_mean(np.abs(pred), 20), -10, axis=-1)
    np.testing.assert_allclose(out, expected, equal_nan=True)
    assert np.nanmin(out) >= 0


def test_threshold_per_trace():
    pred = np.array([[1.0, -3.0], [2.0, 2.0]])
    np.testing.assert_allclose(threshold(pred, 1.5), [3.0, 3.0])
