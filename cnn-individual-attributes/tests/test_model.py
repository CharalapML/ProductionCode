import numpy as np
import pytest

from cnn_individual_attributes.data import prepare_inputs, prepare_targets
from cnn_individual_attributes.model import build_model, compile_model


def test_original_configuration_matches_notebook():
    model = build_model(num_receivers=10, trace_length=8000)
    assert model.input_shape == (None, 10, 1, 8000, 1)
    assert model.output_shape == (None, 10, 1, 8000, 1)
    assert model.count_params() == 9313  # "Total params: 9,313" in the notebook


@pytest.mark.parametrize("trace_length", [0, 39, 100])
def test_rejects_trace_length_not_multiple_of_40(trace_length):
    with pytest.raises(ValueError, match="multiple of 40"):
        build_model(trace_length=trace_length)


def test_rejects_bad_receivers():
    with pytest.raises(ValueError):
        build_model(num_receivers=0)


def test_fit_reduces_loss_and_predicts_finite(synthetic_arrays):
    from tensorflow import keras

    keras.utils.set_random_seed(0)
    inputs, targets = synthetic_arrays
    x, y = prepare_inputs(inputs), prepare_targets(targets)
    model = compile_model(build_model(x.shape[1], x.shape[3]), learning_rate=1e-2)
    history = model.fit(x, y, epochs=15, verbose=0)
    losses = history.history["loss"]
    assert losses[-1] < losses[0]
    pred = model.predict(x, verbose=0)
    assert pred.shape == x.shape
    assert np.isfinite(pred).all()


def test_save_and_load_roundtrip(tmp_path, synthetic_arrays):
    from tensorflow import keras

    x = prepare_inputs(synthetic_arrays[0])
    model = compile_model(build_model(x.shape[1], x.shape[3]))
    path = tmp_path / "model.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    np.testing.assert_allclose(
        loaded.predict(x, verbose=0), model.predict(x, verbose=0), rtol=1e-5, atol=1e-6
    )
