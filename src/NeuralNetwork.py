from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
import copy
import json
import random
import numpy as np
import tensorflow as tf
from itertools import chain
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

try:
    # Keras 3.x (Colab default)
    from keras.utils import register_keras_serializable
except ImportError:
    # Older tf.keras (TF 2.13 and below)
    from tensorflow.keras.saving import register_keras_serializable


def flatten_chain(matrix):
    return list(chain.from_iterable(matrix))

@register_keras_serializable()
def unit_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Loss based on cosine-sine distance for angles in radians."""
    y_true = tf.multiply(y_true, np.pi * 2)
    y_pred = tf.multiply(y_pred, np.pi * 2)

    cos_t, sin_t = tf.cos(y_true), tf.sin(y_true)
    cos_p, sin_p = tf.cos(y_pred), tf.sin(y_pred)

    squared_dist = tf.square(cos_t - cos_p) + tf.square(sin_t - sin_p)
    return tf.reduce_mean(squared_dist)

@register_keras_serializable()
def huber_unit_loss(y_true, y_pred, delta=1.0):
    """
    Huber version of unit_loss for angles in [0,1] (scaled from [0, 2π]).
    
    Args:
        y_true, y_pred: tensors of shape (batch,) or (batch,1) with values in [0, 1]
        delta: threshold between quadratic and linear loss zones (after mapping to sin/cos space)
    """
    # Convert scaled target/pred into radians
    true_angle = y_true * 2.0 * np.pi
    pred_angle = y_pred * 2.0 * np.pi

    # Map to unit circle representation
    true_unit = tf.stack([tf.cos(true_angle), tf.sin(true_angle)], axis=-1)
    pred_unit = tf.stack([tf.cos(pred_angle), tf.sin(pred_angle)], axis=-1)

    # Compute Huber loss between vectors
    error = pred_unit - true_unit
    abs_error = tf.abs(error)
    
    quadratic = 0.5 * tf.square(error)
    linear = delta * (abs_error - 0.5 * delta)
    
    huber_per_dim = tf.where(abs_error <= delta, quadratic, linear)
    
    # Mean over vector dimensions, then mean over batch
    return tf.reduce_mean(tf.reduce_mean(huber_per_dim, axis=-1))

class NeuralNetwork:
    DEFAULT_HPARAMS: Dict[str, Any] = {
        "NUM_UNITS": 30,
        "NUM_LAYERS": 1,
        "DROPOUT": 0.3,
        "LOSS": huber_unit_loss,
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

    @register_keras_serializable()
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
        early_stopping = keras.callbacks.EarlyStopping(
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
        #optimizer = keras.optimizers.Adam(learning_rate=learning_rate_schedule)
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate_schedule, weight_decay=2e-4)

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


    def _uniformize_angles(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_bins: int = 180,
        mode: str = "oversample",   # or "downsample"
        min_per_bin: int | None = None,
        max_per_bin: int | None = None,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make the target angle distribution y ~ Uniform[0,1] by resampling.
        Assumes y is shape (N, 1) scaled to [0,1] (i.e., angle / (2π)).

        - mode="oversample": replicate scarce angles to match dense bins.
        - mode="downsample": drop from dense bins to match scarce bins.
        """
        rng = np.random.default_rng(seed)

        y1 = y.reshape(-1)  # (N,)
        # Guard: keep values in [0,1]
        y1 = np.clip(y1, 0.0, 1.0)
        bins = np.linspace(0.0, 1.0, n_bins + 1, endpoint=True)
        which_bin = np.digitize(y1, bins) - 1
        which_bin = np.clip(which_bin, 0, n_bins - 1)

        # collect indices per bin
        idx_per_bin = [np.where(which_bin == b)[0] for b in range(n_bins)]
        counts = np.array([len(ix) for ix in idx_per_bin])

        # decide target count per bin
        if mode == "oversample":
            target = counts.max() if max_per_bin is None else min(max_per_bin, counts.max())
        elif mode == "downsample":
            nz = counts[counts > 0]
            target = (nz.min() if nz.size > 0 else 0)
            if min_per_bin is not None:
                target = max(target, min_per_bin)
        else:
            raise ValueError("mode must be 'oversample' or 'downsample'")

        # sample indices
        chosen = []
        for ix in idx_per_bin:
            if len(ix) == 0:
                continue
            if mode == "oversample":
                k = target
                draw = rng.choice(ix, size=k, replace=True)
            else:  # downsample
                k = min(target, len(ix))
                draw = rng.choice(ix, size=k, replace=False)
            chosen.append(draw)

        if not chosen:
            # fallback: return originals if something went wrong
            return x, y

        chosen = np.concatenate(chosen, axis=0)
        rng.shuffle(chosen)

        return x[chosen], y[chosen]