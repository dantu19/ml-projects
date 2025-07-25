# Cats vs Dogs Classification

A deep learning project for binary classification of cat and dog images using TensorFlow and the Xception architecture.

## Overview

This project implements a convolutional neural network (CNN) based on the Xception architecture to classify images as either cats or dogs. The model uses transfer learning with pre-trained ImageNet weights and includes data preprocessing, training, and evaluation pipelines.

## Project Structure

```
cats-dogs/
├── data_sources/
│   ├── train/
│   │   ├── cats/          # Training cat images
│   │   └── dogs/          # Training dog images
│   └── test/
│       ├── cats/          # Test cat images
│       └── dogs/          # Test dog images
├── data.py                # Data loading and preprocessing
├── model.py               # Model architecture and training
├── evaluation.py          # Model evaluation utilities
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Features

- **Transfer Learning**: Uses pre-trained Xception model with ImageNet weights
- **Data Preprocessing**: Automatic image resizing and normalization
- **Flexible Architecture**: Configurable model parameters and training options
- **Binary Classification**: Predicts whether an image contains a cat or dog
- **Evaluation Pipeline**: Comprehensive model evaluation on test data

## Requirements

The project requires Python 3.x and the following key dependencies:

- **TensorFlow 2.19.0**: Deep learning framework
- **Keras 3.10.0**: High-level neural network API
- **Pillow 11.3.0**: Image processing
- **NumPy 2.1.3**: Numerical computing
- **Matplotlib 3.10.3**: Plotting and visualization

For a complete list of dependencies, see `requirements.txt`.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cats-dogs
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the dataset:
```bash
python download_data.py
```

**Note**: The dataset is not included in this repository due to its large size (~500MB). Follow the instructions provided by the download script to obtain the Cats vs Dogs dataset from Kaggle or other sources.

## Dataset

The project expects a dataset organized as follows:

- **Training Data**: `data_sources/train/` with subdirectories `cats/` and `dogs/`
- **Test Data**: `data_sources/test/` with subdirectories `cats/` and `dogs/`

Each subdirectory should contain the respective animal images in common formats (JPG, PNG, etc.).

### Getting the Dataset

The dataset is not included in this repository due to its large size. You can obtain it in several ways:

1. **Kaggle Dataset (Recommended)**:
   - Visit: https://www.kaggle.com/datasets/salader/dogs-vs-cats
   - Download the data files
   - Extract and organize files according to the structure above

### Dataset Statistics

- **Training Images**: ~25,000 total (12,500 cats + 12,500 dogs)
- **Test Images**: ~2,500 total (1,250 cats + 1,250 dogs)
- **Image Format**: JPG
- **Average Size**: ~50KB per image
- **Total Dataset Size**: ~500MB

## Usage

### Basic Usage

```python
from cats_dogs import data, model

# Load datasets
train_ds = data.create_dataset(subset="training")
val_ds = data.create_dataset(subset="validation")
test_ds = data.create_dataset(subset="test")

# Create and train model
classifier = model.CatsDogsClassifier()
classifier.train(train_ds, val_ds, epochs=10)

# Make predictions
predictions = classifier.predict(test_ds)
```

### Model Configuration

The `CatsDogsClassifier` class supports various configuration options:

```python
classifier = model.CatsDogsClassifier(
    input_shape=(128, 128, 3),  # Image dimensions
    dense_units=128,            # Dense layer units
    optimizer='adam',           # Optimizer
    freeze_base=True           # Freeze pre-trained layers
)
```

### Data Loading

The `data.create_dataset()` function supports different subsets:

```python
# Training data (80% of train set)
train_ds = data.create_dataset(subset="training")

# Validation data (20% of train set)
val_ds = data.create_dataset(subset="validation")

# Test data
test_ds = data.create_dataset(subset="test")
```

## Model Architecture

The model uses the Xception architecture with the following components:

1. **Input Layer**: Resizes images to 128x128 pixels
2. **Preprocessing**: Applies Xception-specific preprocessing
3. **Base Model**: Pre-trained Xception with ImageNet weights
4. **Global Average Pooling**: Reduces spatial dimensions
5. **Dense Layer**: 128 units with ReLU activation
6. **Output Layer**: Single unit with sigmoid activation for binary classification

## Training

The model is trained using:
- **Loss Function**: Binary cross-entropy
- **Optimizer**: Adam (configurable)
- **Metrics**: Accuracy
- **Validation Split**: 20% of training data

## Evaluation

Run the evaluation script to train and evaluate the model:

```bash
python -c "from cats_dogs.evaluation import evaluate_model; evaluate_model()"
```

## API Reference

### `CatsDogsClassifier`

Main classifier class for cats vs dogs classification.

#### Methods

- `__init__(input_shape, dense_units, optimizer, freeze_base)`: Initialize the classifier
- `build_model()`: Build and compile the model
- `train(train_dataset, validation_dataset, epochs)`: Train the model
- `predict(data, return_probabilities)`: Make predictions

### `data.create_dataset(subset)`

Create TensorFlow datasets for training, validation, or testing.

#### Parameters

- `subset`: One of "training", "validation", or "test"

#### Returns

- TensorFlow dataset with batched and preprocessed images

## Performance

The model typically achieves:
- **Training Accuracy**: ~95%+ after 10 epochs
- **Validation Accuracy**: ~90%+ 
- **Test Accuracy**: ~85-90%
