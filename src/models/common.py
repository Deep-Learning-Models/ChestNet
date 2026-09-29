"""Shared building blocks for all models."""
import keras


def classification_head(x, num_classes: int, dropout: float, name: str = "predictions"):
    """Dropout + Dense with SIGMOID: independent probability per disease (multi-label)."""
    if dropout:
        x = keras.layers.Dropout(dropout, name="head_dropout")(x)
    return keras.layers.Dense(num_classes, activation="sigmoid", dtype="float32", name=name)(x)
