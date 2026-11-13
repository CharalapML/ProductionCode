"""Command-line interface: ``cnn-individual-attributes train|predict``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from cnn_individual_attributes.data import (
    ATTRIBUTE_NAMES,
    DEFAULT_ATTRIBUTE,
    load_dataset,
    prepare_inputs,
    prepare_targets,
    train_test_split,
)
from cnn_individual_attributes.postprocess import smooth_prediction, threshold

log = logging.getLogger("cnn_individual_attributes")


def _train(args: argparse.Namespace) -> int:
    from tensorflow import keras

    from cnn_individual_attributes.model import build_model, compile_model

    if args.seed is not None:
        keras.utils.set_random_seed(args.seed)

    raw_inputs, raw_targets = load_dataset(args.inputs, args.targets)
    x = prepare_inputs(raw_inputs, args.attribute)
    y = prepare_targets(raw_targets)
    x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(
        x, y, test_size=args.test_size, seed=args.seed
    )
    log.info("train events: %s, test events: %s", train_idx.tolist(), test_idx.tolist())

    model = compile_model(build_model(x.shape[1], x.shape[3]), learning_rate=args.learning_rate)
    history = model.fit(
        x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=args.verbose
    )
    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, verbose=0)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.keras")
    np.save(out / "test_predictions.npy", predictions[:, :, 0, :, 0])
    metrics = {
        "attribute": args.attribute,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
        "test_loss": float(test_loss),
        "test_mae": float(test_mae),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("test loss (MSE): %.6g, test MAE: %.6g", test_loss, test_mae)
    log.info("saved model and results to %s", out)
    return 0


def _predict(args: argparse.Namespace) -> int:
    from tensorflow import keras

    raw_inputs, _ = load_dataset(args.inputs)
    x = prepare_inputs(raw_inputs, args.attribute)
    model = keras.models.load_model(args.model)
    expected = tuple(model.input_shape[1:])
    if x.shape[1:] != expected:
        raise ValueError(f"Model expects inputs of shape {expected}, got {x.shape[1:]}")

    predictions = model.predict(x, batch_size=args.batch_size, verbose=0)[:, :, 0, :, 0]
    smoothed = smooth_prediction(predictions, args.window)
    levels = threshold(predictions, args.threshold_factor)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "predictions.npy", predictions)
    np.save(out / "smoothed.npy", smoothed)
    np.save(out / "thresholds.npy", levels)
    log.info("saved predictions %s to %s", predictions.shape, out)

    if args.plot:
        from cnn_individual_attributes.plotting import plot_event

        for n in range(len(predictions)):
            plot_event(
                predictions[n],
                x[n, :, 0, :, 0],
                out / "plots",
                window=args.window,
                threshold_factor=args.threshold_factor,
                prefix=f"event{n:03d}",
            )
        log.info("saved plots to %s", out / "plots")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cnn-individual-attributes", description=__doc__)
    parser.add_argument("-v", "--verbose", action="count", default=0, help="more logging")
    sub = parser.add_subparsers(dest="command", required=True)

    attr_help = f"attribute name {ATTRIBUTE_NAMES} or index (default: {DEFAULT_ATTRIBUTE})"

    t = sub.add_parser("train", help="train a model and evaluate on a held-out split")
    t.add_argument("--inputs", required=True, help=".npy of shape (events, receivers, attributes, trace_length)")
    t.add_argument("--targets", required=True, help=".npy of shape (events, receivers, 1, trace_length)")
    t.add_argument("--attribute", default=DEFAULT_ATTRIBUTE, help=attr_help)
    t.add_argument("--output-dir", required=True, help="directory for model.keras, metrics.json, predictions")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--learning-rate", type=float, default=1e-3)
    t.add_argument("--test-size", type=float, default=0.25)
    t.add_argument("--seed", type=int, default=None, help="seed for the split and weight init")
    t.set_defaults(func=_train)

    p = sub.add_parser("predict", help="run a saved model on new inputs")
    p.add_argument("--model", required=True, help="path to a saved model.keras")
    p.add_argument("--inputs", required=True, help=".npy of shape (events, receivers, attributes, trace_length)")
    p.add_argument("--attribute", default=DEFAULT_ATTRIBUTE, help=attr_help)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--window", type=int, default=20, help="moving-mean window in samples")
    p.add_argument("--threshold-factor", type=float, default=1.5)
    p.add_argument("--plot", action="store_true", help="save per-receiver PNGs (needs matplotlib)")
    p.set_defaults(func=_predict)
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
