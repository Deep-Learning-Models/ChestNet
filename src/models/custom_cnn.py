"""Model 1 – Custom CNN trained from scratch (baseline).

Block = Conv3x3 -> BatchNorm -> ReLU -> Conv3x3 -> BatchNorm -> ReLU -> MaxPool.
No pretrained weights, so comparing it with models 2-4 shows the value of transfer learning.
"""
import keras

from src.models.common import classification_head

GRADCAM_LAYER = "last_conv"


def _conv_bn_relu(x, filters, name):
    x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=name)(x)
    x = keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    return keras.layers.Activation("relu", name=f"{name}_relu")(x)


def build(cfg: dict) -> keras.Model:
    m = cfg["model"]
    size = cfg["data"]["image_size"]
    num_classes = len(cfg["data"]["classes"])

    inputs = keras.Input((size, size, 3), name="image")
    x = keras.layers.Rescaling(1.0 / 255, name="rescale")(inputs)       # 0-255 -> 0-1

    filters = m.get("filters", [32, 64, 128, 256])
    for i, f in enumerate(filters, start=1):
        x = _conv_bn_relu(x, f, name=f"block{i}_conv1")
        last = i == len(filters)
        x = _conv_bn_relu(x, f, name=GRADCAM_LAYER if last else f"block{i}_conv2")
        x = keras.layers.MaxPooling2D(2, name=f"block{i}_pool")(x)

    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(m.get("dense_units", 256), activation="relu", name="dense")(x)
    outputs = classification_head(x, num_classes, m.get("dropout", 0.5))
    return keras.Model(inputs, outputs, name="custom_cnn")
