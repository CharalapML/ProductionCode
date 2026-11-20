# seismic-signal-attributes

Builds a stack of **attributes** (derived signals) from a set of seismic receiver traces
for one event: kurtosis characteristic functions, an automatic onset pick, gain-corrected
signals and onset-aligned auto-/cross-correlations. The attributes are intended as
inputs to a machine-learning model that locates the depth phase in each trace. Their
order matches the input arrays used by
[`cnn-individual-attributes`](../cnn-individual-attributes/), where indices 0–4 are
the same five attributes.

This is the production version of the Colab notebook
`Seismic_Signal_Derivatives-Kurtosis_Auto_and_XcorrelationsGainFuncsCepstrum and array build.ipynb`,
written while volunteering at the International Seismological Centre (ISC), with
thanks to Dr Tom Garth for his guidance. The original notebook, with outputs cleared,
is in [`notebooks/`](notebooks/).

The kurtosis functions follow C. Baillard, W. Crawford, V. Ballu, C. Hibert and
A. Mangeney (2014), *An Automatic Kurtosis-Based P- and S-Phase Picker Designed for
Local Seismic Networks*, BSSA 104(1), 394–409.

## What it does

For each trace (any format ObsPy reads: miniSEED, SAC, ...):

1. **Detrend and scale.** Removes a spline trend (`obspy.signal.detrend.spline`,
   order 2, a knot every 1000 samples) and scales to max |x| = 1 (`SigSplineScale`).
2. **Characteristic functions.** F1cf is the kurtosis over a trailing 100-sample window.
   F2cf accumulates F1cf's rises, F3cf removes F2cf's linear trend, F4cf marks where
   F3cf rises, and F4cfgrad is the positive part of F4cf's gradient. F4cfgrad_gtmean
   keeps the F4cfgrad spikes above 8 × the mean positive value.
3. **Onset.** Up to 7 samples back from the F1cf peak, each F4cfgrad_gtmean spike
   proposes the nearest earlier sample where F4cf is zero, looking back at most 7
   samples. The onset is the latest proposal. Later F4cf events after the peak are
   reported in the metadata.
4. **Gains.** `SigSplineScalePreSetGain` applies a fixed gain shape around the onset.
   `SigSplineScaleAmpRec` divides the signal by an `a·exp(−b(i − onset)) + c` curve
   fitted to its post-onset peak-amplitude trend.
5. **Correlations.** Each trace is correlated with every other trace, and the result
   is shifted so its peak lies on the trace's own onset. The *impulse* correlations
   slide the 60 samples (3 s at 20 Hz) from an onset along the trace, which is zeroed
   before its own onset.

Output: one array of shape `(traces, 12, samples)`. Traces shorter than the longest are
zero-padded at the end.

| index | name                       | description |
|-------|----------------------------|-------------|
| 0     | `SigSplineScale`           | detrended signal scaled to max \|x\| = 1 |
| 1     | `F1cf`                     | kurtosis characteristic function (÷ 4) |
| 2     | `F4cf`                     | rising parts of F3cf (× 5) |
| 3     | `F4cfgrad`                 | positive gradient of F4cf |
| 4     | `F4cfgrad_gtmean`          | F4cfgrad above 8 × its positive mean |
| 5     | `env_pos`                  | 8-sample mean of \|x\| × 1.5 (approximate envelope) |
| 6     | `SigSplineScalePreSetGain` | signal × fixed gain: 1.01^(i−onset) before, 1 for 200 samples, then rising to 2 |
| 7     | `XcorrShiftAdded`          | sum of onset-aligned full cross-correlations with the other traces |
| 8     | `AllImpACorr`              | the trace's own 60-sample onset impulse slid along the trace |
| 9     | `AXcorrShiftAdded`         | as 7, plus the trace's own aligned autocorrelation |
| 10    | `SigSplineScaleAmpRec`     | signal ÷ fitted exponential decay |
| 11    | `AllImpXcorrAdded`         | sum of every *other* trace's onset impulse slid along this trace |

The window lengths are in samples and were chosen for 20 Hz data. By default, traces
at any other rate are skipped with a warning.

## Install

Requires Python 3.12 (other versions are rejected by `pip install`).

```bash
cd seismic-signal-attributes
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[plot]"        # add ,dev for the test dependencies
```

## Usage

### Command line

```bash
seismic-signal-attributes build data/event1/*.BHZ --output-dir outputs/event1 --plot
```

This writes:

- `attributes.npy`, shape `(traces, 12, samples)`, in input file order (after skipped
  traces are removed).
- `metadata.json`, with `attribute_names` and one entry per trace: file, trace id,
  sampling rate, sample count, start time, onset (sample, seconds from start, UTC),
  the F4cfgrad threshold, later F4cf onsets (samples, and seconds after the onset), and
  the fitted gain parameters `a`, `b`, `c`.
- With `--plot`: `plots/NNN_<file>.png`, all 12 attributes around the onset.

`--sampling-rate 40` processes 40 Hz traces instead, and `--sampling-rate 0` keeps every
trace. If a trace can't be processed (for example, no onset is found), the command exits
with status 2 and an error naming the file.

### Python

```python
from seismic_signal_attributes import ATTRIBUTE_NAMES, build_attribute_array, load_traces

traces = load_traces(["IC.HIA.00.BHZ", "II.AAK.00.BHZ", "IC.XAN.00.BHZ"])
array, metadata = build_attribute_array(traces)
cnn_inputs = array[:, :5]          # the five attributes cnn-individual-attributes uses
```

The individual steps are public too: `spline_scale`, `characteristic_functions`,
`detect_onset`, `detect_subsequent_onsets`, `preset_gain`, `amplitude_recovery`,
`aligned_correlation_sum`, `impulse_correlation`, and `real_cepstrum` (the notebook's
exploratory cepstrum, which isn't part of the array).

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests use synthetic events, so no real data is needed. The arrivals are known, and
the traces are written to miniSEED for the I/O and CLI tests. `tests/reference.py` is a
line-by-line transcription of the notebook's loops. The vectorised code is checked
against it for the characteristic functions, the onset picks, the gains and the
correlations.

## Differences from the notebook

- Colab and Google Drive paths are replaced by command-line arguments. No data is bundled.
- The loops are vectorised, and the full cross-correlations use FFTs
  (`scipy.signal.correlate`). Six 110,000-sample traces take well under a second.
- **Fixed: inputs were modified in place.** The impulse correlations zeroed the
  pre-onset part of the shared signal arrays, including attribute 0 of the final
  array. So the results depended on trace order. Every function now works on copies,
  and attribute 0 is the unzeroed signal.
- **Fixed: wrap-around in the aligned correlations.** `np.roll` wrapped the shifted
  correlation, and the tail-zeroing was off by two samples. Vacated samples are now
  zero. Results are identical wherever no wrap occurred.
- Each trace is processed from a single read of its own file. In the notebook the
  correlations used a separately built list that could misalign after traces were
  dropped for their sampling rate. The trend fit also used another trace's sample axis.
- Traces of different lengths are zero-padded. The notebook allocated a padded array
  but then failed to fill it.
- The last F4cfgrad sample is now clipped at zero like all the others.
- Constant windows give F1cf = 0 instead of NaN. The onset search no longer wraps to
  the end of the trace through negative indices. A missing onset, an impulse that
  doesn't fit, or a non-positive fitted gain raises an error that names the trace.
- Later F4cf onsets are returned as sorted unique sample indices.
- Code with no effect on the output was dropped: the unused "late F1cf peak" search,
  the reset-time and zeroed-signal columns, and the commented-out skew attributes. The
  notebook's check that warns when the F1cf peak is more than 30 s before the largest
  amplitude is kept, as a log warning.
- The notebook's `featuresDict` labelled index 8 `AllImpXcorrAdded`, but it holds the
  impulse *auto*correlation. It is named `AllImpACorr` here.
- Exploratory plots are not included. `--plot` writes one summary figure per trace.
