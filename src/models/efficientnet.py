"""Model 3 – EfficientNet-B0 / B3 (compound scaling of depth, width and resolution).

🧠 The idea in one picture
    Older CNNs grew in ONE direction (ResNet: deeper; WideResNet: wider; or bigger images).
    EfficientNet grows all three TOGETHER with one knob, phi:
        depth  x 1.2^phi   (more layers)
        width  x 1.1^phi   (more channels per layer)
        resolution x 1.15^phi (bigger input images)
    B0 is the small base network (224 px, ~4.0 M backbone params).
    B3 is B0 scaled up (native 300 px, ~10.8 M backbone params).

    Its building block is the MBConv ("inverted residual") block:
        1x1 expand -> 3x3/5x5 DEPTHWISE conv -> Squeeze-and-Excitation -> 1x1 project (+ skip)
    Depthwise convs + SE give ResNet-level accuracy with ~5-10x fewer parameters.

📥 Input contract (same as every ChestNet model)
    float32, shape (size, size, 3), pixel range 0-255.
    Keras EfficientNet has its OWN Rescaling + Normalization layers inside, so we must NOT
    divide by 255 beforehand (that would normalise twice and the pretrained features break).

🔧 Fine-tuning choice: frozen BatchNorm (model.freeze_bn, default true)
    BatchNorm layers keep running statistics learned on ImageNet. With small batches of
    grey X-rays those statistics get overwritten by noisy estimates and accuracy drops.
    Keras' official EfficientNet fine-tuning guide therefore keeps BN layers frozen
    (inference mode) while all conv weights are trained. This is an architecture-specific
    setting (allowed under model.*) and is logged in config_used.yaml.
"""
import keras

from src.models.common import classification_head

GRADCAM_LAYER = "top_activation"     # last conv feature map (7x7 at 224 px) – same name in B0..B7

_VARIANTS = {
    "B0": (keras.applications.EfficientNetB0, 224),
    "B1": (keras.applications.EfficientNetB1, 240),
    "B2": (keras.applications.EfficientNetB2, 260),
    "B3": (keras.applications.EfficientNetB3, 300),
}


def native_resolution(variant: str) -> int:
    """Image size the variant was designed and pretrained for (B0 224, B3 300)."""
    return _VARIANTS[str(variant).upper()][1]


def _freeze_batchnorm(backbone: keras.Model) -> int:
    """Put every BatchNorm layer in inference mode (keeps ImageNet statistics). Returns the count."""
    n = 0
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
            n += 1
    return n


def build(cfg: dict) -> keras.Model:
    m = cfg["model"]
    size = cfg["data"]["image_size"]
    num_classes = len(cfg["data"]["classes"])
    variant = str(m.get("variant", "B0")).upper()
    if variant not in _VARIANTS:
        raise ValueError(f"EfficientNet variant '{variant}' not supported. Choose from {list(_VARIANTS)}")
    constructor, _ = _VARIANTS[variant]

    inputs = keras.Input((size, size, 3), name="image")          # raw 0-255 pixels
    backbone = constructor(
        include_top=False,                                        # drop the 1000-class ImageNet head
        weights="imagenet" if m.get("pretrained", True) else None,
        input_tensor=inputs,                                      # backbone layers live in OUR graph (Grad-CAM needs this)
    )

    # Safety check: the built-in normalisation must be there, otherwise inputs would be 0-255 raw.
    layer_types = {type(l).__name__ for l in backbone.layers[:5]}
    assert "Rescaling" in layer_types, "EfficientNet lost its built-in Rescaling layer – check the Keras version"

    backbone.trainable = not m.get("freeze_backbone", False)
    # Freeze BN ONLY with pretrained weights: without them the BN statistics are just the
    # initial values (mean 0, var 1), and freezing those stops the network from learning.
    if backbone.trainable and m.get("pretrained", True) and m.get("freeze_bn", True):
        _freeze_batchnorm(backbone)

    x = keras.layers.GlobalAveragePooling2D(name="gap")(backbone.output)   # (7,7,C) -> (C,)
    outputs = classification_head(x, num_classes, m.get("dropout", 0.3))   # sigmoid: one prob per disease
    return keras.Model(inputs, outputs, name=f"efficientnet_{variant.lower()}")
