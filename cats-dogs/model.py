"""Module for a binary classifier using Xception architecture for cats vs dogs classification."""
import tensorflow as tf
from typing import Tuple

class CatsDogsClassifier:
    """Xception-based binary classifier for cats vs dogs."""
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (128, 128, 3),
        dense_units: int = 128,
        optimizer: str = 'adam',
        freeze_base: bool = True
    ):
        """
        Initialize the cats vs dogs classifier.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            dense_units: Number of units in the dense layer
            optimizer: Optimizer for training
            freeze_base: Whether to freeze the base Xception model
        """
        self.input_shape = input_shape
        self.dense_units = dense_units
        self.optimizer = optimizer
        self.freeze_base = freeze_base
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """Build the Xception-based model."""

        base_model = tf.keras.applications.Xception(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        base_model.trainable = not self.freeze_base

        model = tf.keras.Sequential([
            tf.keras.layers.Resizing(self.input_shape[0], self.input_shape[1]),
            tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(self.dense_units, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=self.optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(
        self,
        train_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        epochs: int = 5,
    ):
        """
        Train the model on the provided dataset.

        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset (optional)
            epochs: Number of training epochs
        """
        self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            verbose=1
        )