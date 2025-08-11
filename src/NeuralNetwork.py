from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
import copy
import json
import random
import numpy as np
import tensorflow as tf
from itertools import chain
from tensorflow import keras
from keras import callbacks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def flatten_chain(matrix):
    return list(chain.from_iterable(matrix))

@keras.saving.register_keras_serializable()
def unit_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Loss based on cosine-sine distance for angles in radians."""
    y_true = tf.multiply(y_true, np.pi * 2)
    y_pred = tf.multiply(y_pred, np.pi * 2)

    cos_t, sin_t = tf.cos(y_true), tf.sin(y_true)
    cos_p, sin_p = tf.cos(y_pred), tf.sin(y_pred)

    squared_dist = tf.square(cos_t - cos_p) + tf.square(sin_t - sin_p)
    return tf.reduce_mean(squared_dist)

class NeuralNetwork:
    DEFAULT_HPARAMS: Dict[str, Any] = {
        "NUM_UNITS": 30,
        "NUM_LAYERS": 1,
        "DROPOUT": 0.3,
        "LOSS": unit_loss,
        "OPTIMIZER": "adam",
        "VALIDATE_SPLIT": 0.2,
        "EPOCHS": 200,
        "LEARNING_RATE": 1e-3,
        "BATCH_SIZE": 32,
        "PATIENCE": 50,  # Recommended: ≈ EPOCHS/4 for small datasets, EPOCHS/10 for large datasets
        "SEED": 42
    }

    def __init__(self, verbose: bool = True) -> None:
        self.hparams = dict(self.DEFAULT_HPARAMS)
        self.metric_accuracy = "mse"
        self.verbose = verbose
        self.model: Optional[keras.Model] = None
        self.history: Optional[keras.callbacks.History] = None
        self.offset: float = 0.0
        self.range: float = 2 * np.pi
        self.input_shape: Tuple[Optional[int], int] = (None, 1)
        self.output_shape: int = 1
        tf.config.run_functions_eagerly(True)

    @keras.saving.register_keras_serializable()
    def normalize(self, params: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Normalize angles in [0, 2π] to [0, 1] range without altering the shape.
        Works for both single-output (n, 1) and multi-output (n, k) datasets.
        """
        params_arr = np.array(params, dtype=np.float64)  # Keep original shape

        # Wrap values to [0, 2π]
        params_arr = np.where(params_arr > 2 * np.pi, params_arr - 2 * np.pi, params_arr)
        params_arr = np.where(params_arr < 0, params_arr + 2 * np.pi, params_arr)

        # Store normalization constants for inverse transformation
        self.offset = 0.0
        self.range = 2 * np.pi

        # Normalize to [0, 1]
        normalized = (params_arr - self.offset) / self.range

        if self.verbose:
            print(f"range : {self.range}")
            print(f"offset : {self.offset}")
            print(f"shape preserved: {normalized.shape}")

        return normalized

    def invert_normalize(self, params: np.ndarray) -> np.ndarray:
        """
        Convert normalized [0, 1] data back to [0, 2π] scale.
        """
        return params * self.range + self.offset

    def _build_model(self) -> None:
        """Build the Keras model."""
        layers: List[keras.layers.Layer] = [
            tf.keras.layers.Normalization(input_shape=self.input_shape[1:]),
            keras.layers.Dense(self.hparams["NUM_UNITS"], activation="relu"),
        ]
        for _ in range(self.hparams["NUM_LAYERS"]):
            layers.append(
                keras.layers.Dense(self.hparams["NUM_UNITS"], activation="relu")
            )
        layers.append(keras.layers.Dropout(self.hparams["DROPOUT"]))
        layers.append(
            keras.layers.Dense(
                self.output_shape,
                activation="sigmoid",
                kernel_initializer="normal",
            )
        )

        self.model = keras.Sequential(layers)
        self.model.build(self.input_shape)

        if self.verbose:
            self.model.summary()

    def _compile_and_train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[float, keras.callbacks.History]:
        if not self.model:
            raise RuntimeError("Model must be built before training.")

        cp_callback = keras.callbacks.ModelCheckpoint(
            filepath="training_logs/cp.weights.h5", save_weights_only=True, verbose=1
        )
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=self.hparams["PATIENCE"],
            restore_best_weights=True,
        )
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.hparams["LEARNING_RATE"],
            decay_steps=10000,
            decay_rate=0.96,
        )
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate_schedule)

        self.model.compile(
            optimizer=optimizer,
            loss=self.hparams["LOSS"],
            metrics=["mean_squared_error"],
        )

        history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            batch_size=self.hparams["BATCH_SIZE"],
            epochs=self.hparams["EPOCHS"],
            callbacks=[cp_callback, early_stopping],
            verbose=1 if self.verbose else 0,
        )

        _, accuracy = self.model.evaluate(x_val, y_val, verbose=0)

        # Save history as JSON
        os.makedirs("training_logs", exist_ok=True)
        with open("training_logs/history.json", "w") as f:
            json.dump(history.history, f, indent=4)

        return accuracy, history

    def train_model(
        self,
        run_dir: str,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> keras.callbacks.History:
        with tf.summary.create_file_writer(run_dir).as_default():
            accuracy, history = self._compile_and_train(*data)
            tf.summary.scalar(self.metric_accuracy, accuracy, step=1)
        return history

    def run(
        self,
        x_in: Sequence[Sequence[float]],
        y_in: Sequence[Sequence[float]],
        hparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Run full training process with improved validation split."""
        if hparams:
            self.hparams.update(hparams)

        # Set seeds for reproducibility
        seed = self.hparams["SEED"]
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        x = copy.deepcopy(x_in)
        y = copy.deepcopy(y_in)
        y = self.normalize(y)

        # Use sklearn train_test_split for consistent coverage
        x_train, x_val, y_train, y_val = train_test_split(
            np.array(x),
            np.array(y),
            test_size=self.hparams["VALIDATE_SPLIT"],
            random_state=seed,
            shuffle=True,
        )

        x_width = x_val.shape[1] if x_val.ndim > 1 else 1
        self.input_shape = (None, x_width)
        self.output_shape = y_train.shape[1]

        self._build_model()
        self.history = self.train_model(
            "logs/hparam_tuning/test", (x_train, y_train, x_val, y_val)
        )
        self.plot_history()

    def plot_history(self) -> None:
        if not self.history:
            raise RuntimeError("No training history found to plot.")

        plt.plot(self.history.history["val_loss"])
        plt.plot(self.history.history["loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.yscale("log")
        plt.legend(["Validation", "Training"], loc="upper left")
        plt.show()

    def load_model(self, filepath: str) -> None:
        self.model = tf.keras.models.load_model(filepath)
