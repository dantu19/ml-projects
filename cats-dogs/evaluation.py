"""Evaluation of the cats vs dogs model."""
from cats_dogs import data, model

def evaluate_model() -> None:
    """Evaluate the cats vs dogs model."""
    # Load all datasets
    train_ds = data.create_dataset(subset="training")
    val_ds = data.create_dataset(subset="validation")
    test_ds = data.create_dataset(subset="test")

    # Create the model
    cd_model = model.CatsDogsClassifier()
    cd_model = cd_model.build_model()

    # Train the model
    cd_model.train(
        train_dataset=train_ds,
        validation_dataset=val_ds,
    )

    # Evaluate the model
    cd_model.evaluate(
        test_dataset=test_ds,
        verbose=0,
    )
