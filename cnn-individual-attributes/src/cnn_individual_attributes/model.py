"""The 3D convolutional encoder-decoder from the original notebook.

Convolutions and pooling act only along the trace (time) axis; receivers are
processed independently. The encoder downsamples time by 2*2*2*5 = 40 and the
decoder upsamples by 5*2*4 = 40, with skip connections (concatenation) at each
scale, so ``trace_length`` must be a multiple of 40.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

DOWNSAMPLE_FACTOR = 40


def build_model(num_receivers: int = 10, trace_length: int = 8000) -> keras.Model:
    """Build the network for inputs of shape ``(num_receivers, 1, trace_length, 1)``."""
    if num_receivers < 1:
        raise ValueError(f"num_receivers must be >= 1, got {num_receivers}")
    if trace_length < DOWNSAMPLE_FACTOR or trace_length % DOWNSAMPLE_FACTOR:
        raise ValueError(
            f"trace_length must be a positive multiple of {DOWNSAMPLE_FACTOR}, got {trace_length}"
        )

    inputs = keras.Input((num_receivers, 1, trace_length, 1))

    # Encoder
    x = layers.Conv3D(32, kernel_size=(1, 1, 2), padding="same")(inputs)
    x = layers.MaxPool3D(pool_size=(1, 1, 2), strides=(1, 1, 2))(x)

    x1 = layers.Conv3D(16, kernel_size=(1, 1, 5), padding="same", activation="relu")(x)
    x1 = layers.MaxPool3D(pool_size=(1, 1, 2), strides=(1, 1, 2))(x1)

    x2 = layers.Conv3D(8, kernel_size=(1, 1, 2), padding="same", activation="relu")(x1)
    x2 = layers.MaxPool3D(pool_size=(1, 1, 2), strides=(1, 1, 2))(x2)

    x3 = layers.Conv3D(
        8, kernel_size=(1, 1, 5), strides=(1, 1, 5), padding="same", activation="relu"
    )(x2)

    # Decoder with skip connections
    y3 = layers.Conv3DTranspose(8, kernel_size=(1, 1, 5), padding="same")(x3)
    y3 = layers.Concatenate()([y3, x3])
    y3 = layers.UpSampling3D((1, 1, 5))(y3)

    y2 = layers.Conv3DTranspose(8, kernel_size=(1, 1, 2), padding="same")(y3)
    y2 = layers.Concatenate()([y2, x2])
    y2 = layers.UpSampling3D((1, 1, 2))(y2)

    y1 = layers.Conv3DTranspose(16, kernel_size=(1, 1, 5), padding="same")(y2)
    y1 = layers.Concatenate()([y1, x1])
    y1 = layers.UpSampling3D((1, 1, 4))(y1)

    y = layers.Conv3DTranspose(32, kernel_size=(1, 1, 4), padding="same")(y1)
    # Kept from the original notebook: with a single attribute this pooling is an identity.
    y = layers.MaxPool3D(pool_size=(1, 1, 1), strides=(1, 5, 1))(y)
    outputs = layers.Conv3D(1, kernel_size=(1, 1, 1), padding="same")(y)

    return keras.Model(inputs=inputs, outputs=outputs, name="cnn_individual_attributes")


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    """Compile with Adam + MSE, as in the notebook (MAE replaces 'accuracy',
    which is not meaningful for a regression target)."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model
