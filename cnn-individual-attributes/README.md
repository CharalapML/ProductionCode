# cnn-individual-attributes

A 3D convolutional encoder–decoder that takes **one seismic attribute** per receiver
trace and predicts a target trace of the same length. This is the production version of
the Colab notebook `60_NN_IndividualCubes_Individual Atrributes_20 01_2022.ipynb`
(ISC 2020 project). The original notebook, with outputs cleared, is in
[`notebooks/`](notebooks/).

## What it does

1. Loads a 4D input array `(events, receivers, attributes, trace_length)` and a target
   array `(events, receivers, 1, trace_length)` from `.npy` files. NaNs in the targets
   are set to 0; inputs must be finite.
2. Selects a single attribute (default `F4cfgrad_gtmean`) and adds a channel axis.
3. Trains the network (Adam, MSE loss) on a random 75/25 split of events.
4. Post-processes predictions per receiver: `|prediction|`, a moving mean
   (window 20, centred by rolling back half a window), and a threshold of
   `1.5 × mean(|prediction|)`.

Attributes, in input-array order (axis 2):

| index | name              | description                    |
|-------|-------------------|--------------------------------|
| 0     | `signal`          | original splined, scaled signal |
| 1     | `F1cf`            | F1 characteristic function      |
| 2     | `F4cf`            | F4 characteristic function      |
| 3     | `F4cfgrad`        | gradient of F4cf                |
| 4     | `F4cfgrad_gtmean` | F4cfgrad above its mean         |

### Network

Convolutions and pooling act only along the time axis, so each receiver is processed
independently. The encoder downsamples time by 2·2·2·5 = 40; the decoder upsamples by
5·2·4 = 40 with concatenation skip connections at each scale. **`trace_length` must be a
multiple of 40.** With the original shapes (10 receivers × 8000 samples) the model has
9,313 parameters, identical to the notebook.

## Install

Requires Python 3.12 (other versions are rejected by `pip install`).

```bash
cd cnn-individual-attributes
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[plot]"        # add ,dev for the test dependencies
```

## Usage

### Command line

```bash
cnn-individual-attributes train \
    --inputs data/4Darray_2_Ordered_in.npy \
    --targets data/4Darray_2_Ordered_out.npy \
    --attribute F4cfgrad_gtmean \
    --epochs 10 --seed 42 \
    --output-dir runs/f4cfgrad_gtmean
```

Writes `model.keras`, `metrics.json` (train/test event indices, loss history, test MSE
and MAE) and `test_predictions.npy` to the output directory.

```bash
cnn-individual-attributes predict \
    --model runs/f4cfgrad_gtmean/model.keras \
    --inputs data/new_events_in.npy \
    --output-dir runs/f4cfgrad_gtmean/predict \
    --plot
```

Writes `predictions.npy` `(events, receivers, trace_length)`, `smoothed.npy`
(same shape; the first `window − 1` samples before the roll are NaN), `thresholds.npy`
`(events, receivers)`, and with `--plot` one PNG per receiver. Run with `--help` for
every option.

### Python

```python
from cnn_individual_attributes import (
    build_model, compile_model, load_dataset, prepare_inputs, prepare_targets,
    smooth_prediction, threshold, train_test_split,
)

inputs, targets = load_dataset("in.npy", "out.npy")
x, y = prepare_inputs(inputs, "F4cfgrad_gtmean"), prepare_targets(targets)
x_train, x_test, y_train, y_test, _, _ = train_test_split(x, y, test_size=0.25, seed=42)

model = compile_model(build_model(num_receivers=x.shape[1], trace_length=x.shape[3]))
model.fit(x_train, y_train, epochs=10)

pred = model.predict(x_test)[:, :, 0, :, 0]       # (events, receivers, trace_length)
smoothed, levels = smooth_prediction(pred, window=20), threshold(pred, 1.5)
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests use small synthetic arrays (no real data needed) and cover data loading and
validation, the model's shapes and parameter count, training, save/load, the
post-processing maths, and the CLI end to end.

## Differences from the notebook

- Colab/Google Drive paths replaced by command-line arguments; no data is bundled.
- The metric is MAE instead of `accuracy`, which has no meaning for a regression target.
- Targets get an explicit channel axis so they match the model output exactly
  (Keras previously squeezed the prediction implicitly).
- The train/test split can be seeded (`--seed`) for reproducible runs.
- The moving mean is implemented in NumPy (same result as `bottleneck.move_mean`), so
  `bottleneck`, `scipy` and `scikit-learn` are no longer dependencies.
- Exploratory plots and the `visualkeras` / `plot_model` diagrams are not included;
  `--plot` reproduces the final per-receiver diagnostic plot.
