"""Data loading and preprocessing for Cats vs Dogs dataset."""
from typing import Literal
import tensorflow as tf


def create_dataset(subset: Literal["training", "test", "validation"]):
    """Create dataset for Cats vs Dogs classification."""

    data_dir = './data_sources/test' if subset == "test" else './data_sources/train'

    if subset == "test":
        kwargs = {
            'image_size': (128, 128),
            'batch_size': 32,
            'shuffle': False,
        }
    else:
        kwargs = {
            'image_size': (128, 128),
            'batch_size': 32,
            'validation_split': 0.2,
            'subset': subset,
            'seed': 123,
        }

    return tf.keras.utils.image_dataset_from_directory(data_dir, **kwargs)
