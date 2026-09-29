"""Model 2 – ResNet50 with ImageNet weights, fine-tuned on chest X-rays.

Residual (skip) connections let gradients flow through 50 layers without vanishing.
"""
import keras
from keras import ops

from src.models.common import classification_head

GRADCAM_LAYER = "conv5_block3_out"
_IMAGENET_BGR_MEAN = [103.939, 116.779, 123.68]


def _caffe_preprocess(x):
    """ResNet50's expected input: RGB -> BGR, subtract ImageNet channel means (no scaling)."""
    x = x[..., ::-1]
    return x - ops.convert_to_tensor(_IMAGENET_BGR_MEAN, dtype=x.dtype)


def build(cfg: dict) -> keras.Model:
    m = cfg["model"]
    size = cfg["data"]["image_size"]
    num_classes = len(cfg["data"]["classes"])

    inputs = keras.Input((size, size, 3), name="image")
    x = keras.layers.Lambda(_caffe_preprocess, name="resnet_preprocess")(inputs)
    backbone = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet" if m.get("pretrained", True) else None,
        input_tensor=x,
    )
    backbone.trainable = not m.get("freeze_backbone", False)

    x = keras.layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    outputs = classification_head(x, num_classes, m.get("dropout", 0.3))
    return keras.Model(inputs, outputs, name="resnet50")
