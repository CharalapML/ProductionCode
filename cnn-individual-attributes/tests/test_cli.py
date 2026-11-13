import json

import numpy as np

from cnn_individual_attributes.cli import main


def test_train_then_predict_end_to_end(tmp_path, synthetic_files):
    inputs, targets = synthetic_files
    run = tmp_path / "run"
    rc = main([
        "train", "--inputs", str(inputs), "--targets", str(targets),
        "--output-dir", str(run), "--epochs", "2", "--seed", "0",
    ])
    assert rc == 0
    assert (run / "model.keras").exists()
    metrics = json.loads((run / "metrics.json").read_text())
    assert len(metrics["train_indices"]) == 3 and len(metrics["test_indices"]) == 1
    assert len(metrics["history"]["loss"]) == 2
    assert np.load(run / "test_predictions.npy").shape == (1, 3, 80)

    pred_dir = tmp_path / "pred"
    rc = main([
        "predict", "--model", str(run / "model.keras"), "--inputs", str(inputs),
        "--output-dir", str(pred_dir), "--window", "10", "--plot",
    ])
    assert rc == 0
    assert np.load(pred_dir / "predictions.npy").shape == (4, 3, 80)
    assert np.load(pred_dir / "smoothed.npy").shape == (4, 3, 80)
    assert np.load(pred_dir / "thresholds.npy").shape == (4, 3)
    assert len(list((pred_dir / "plots").glob("*.png"))) == 4 * 3


def test_predict_rejects_wrong_shape(tmp_path, synthetic_files):
    inputs, targets = synthetic_files
    run = tmp_path / "run"
    main(["train", "--inputs", str(inputs), "--targets", str(targets),
          "--output-dir", str(run), "--epochs", "1"])
    other = tmp_path / "other.npy"
    np.save(other, np.zeros((1, 3, 5, 120), dtype=np.float32))
    rc = main(["predict", "--model", str(run / "model.keras"), "--inputs", str(other),
               "--output-dir", str(tmp_path / "p")])
    assert rc == 2


def test_missing_file_returns_error_code(tmp_path):
    rc = main(["predict", "--model", "m.keras", "--inputs", str(tmp_path / "missing.npy"),
               "--output-dir", str(tmp_path)])
    assert rc == 2
