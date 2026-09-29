"""Model 3 – EfficientNet-B0/B3 (compound scaling of depth, width and resolution).

Keras EfficientNet normalises internally, so it takes 0-255 images directly.
"""
import keras

from src.models.common import classification_head

GRADCAM_LAYER = "top_activation"

_VARIANTS = {
    "B0": keras.applications.EfficientNetB0,
    "B3": keras.applications.EfficientNetB3,
}


def build(cfg: dict) -> keras.Model:
    m = cfg["model"]
    size = cfg["data"]["image_size"]
    num_classes = len(cfg["data"]["classes"])
    variant = str(m.get("variant", "B0")).upper()

    inputs = keras.Input((size, size, 3), name="image")
    backbone = _VARIANTS[variant](
        include_top=False,
        weights="imagenet" if m.get("pretrained", True) else None,
        input_tensor=inputs,
    )
    backbone.trainable = not m.get("freeze_backbone", False)

    x = keras.layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    outputs = classification_head(x, num_classes, m.get("dropout", 0.3))
    return keras.Model(inputs, outputs, name=f"efficientnet_{variant.lower()}")
