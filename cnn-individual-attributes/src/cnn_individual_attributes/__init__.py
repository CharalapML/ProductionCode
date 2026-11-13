"""3D CNN encoder-decoder trained on one seismic attribute per receiver trace.

Production version of the Colab notebook
``60_NN_IndividualCubes_Individual Atrributes_20 01_2022.ipynb``.
"""

from cnn_individual_attributes.data import (
    ATTRIBUTE_NAMES,
    load_dataset,
    prepare_inputs,
    prepare_targets,
    select_attribute,
    train_test_split,
)
from cnn_individual_attributes.model import DOWNSAMPLE_FACTOR, build_model, compile_model
from cnn_individual_attributes.postprocess import moving_mean, smooth_prediction, threshold

__version__ = "1.0.0"

__all__ = [
    "ATTRIBUTE_NAMES",
    "DOWNSAMPLE_FACTOR",
    "build_model",
    "compile_model",
    "load_dataset",
    "moving_mean",
    "prepare_inputs",
    "prepare_targets",
    "select_attribute",
    "smooth_prediction",
    "threshold",
    "train_test_split",
]
