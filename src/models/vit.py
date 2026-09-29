"""Model 4 – Vision Transformer (ViT).

The image is cut into 16x16 patches, each patch becomes a token, and self-attention
lets every patch look at every other patch (global context across both lungs).

pretrained: true  -> ViT-B/16 ImageNet weights from keras-hub (recommended)
pretrained: false -> a smaller ViT built from scratch below (data hungry)
"""
import keras
from keras import layers, ops

from src.models.common import classification_head

GRADCAM_LAYER = None  # Grad-CAM is used for the CNN models only


class ClassToken(layers.Layer):
    """Learnable [CLS] token prepended to the patch sequence."""

    def build(self, input_shape):
        self.cls = self.add_weight(shape=(1, 1, input_shape[-1]), initializer="zeros", name="cls")

    def call(self, x):
        cls = ops.broadcast_to(self.cls, (ops.shape(x)[0], 1, ops.shape(x)[-1]))
        return ops.concatenate([cls, x], axis=1)


class PositionEmbedding(layers.Layer):
    """Learnable position embedding so the model knows where each patch came from."""

    def build(self, input_shape):
        self.pos = self.add_weight(shape=(1, input_shape[1], input_shape[2]),
                                   initializer=keras.initializers.RandomNormal(stddev=0.02), name="pos")

    def call(self, x):
        return x + self.pos


def _encoder_block(x, m, i):
    h = layers.LayerNormalization(epsilon=1e-6, name=f"enc{i}_ln1")(x)
    h = layers.MultiHeadAttention(num_heads=m["num_heads"], key_dim=m["hidden_dim"] // m["num_heads"],
                                  dropout=m.get("dropout", 0.1), name=f"enc{i}_mhsa")(h, h)
    x = layers.Add(name=f"enc{i}_add1")([x, h])
    h = layers.LayerNormalization(epsilon=1e-6, name=f"enc{i}_ln2")(x)
    h = layers.Dense(m["mlp_dim"], activation="gelu", name=f"enc{i}_mlp1")(h)
    h = layers.Dropout(m.get("dropout", 0.1))(h)
    h = layers.Dense(m["hidden_dim"], name=f"enc{i}_mlp2")(h)
    return layers.Add(name=f"enc{i}_add2")([x, h])


def _scratch_vit(x, m):
    p = m.get("patch_size", 16)
    x = layers.Conv2D(m["hidden_dim"], p, strides=p, name="patch_embedding")(x)   # patches -> tokens
    x = layers.Reshape((-1, m["hidden_dim"]), name="tokens")(x)
    x = ClassToken(name="class_token")(x)
    x = PositionEmbedding(name="position_embedding")(x)
    x = layers.Dropout(m.get("dropout", 0.1))(x)
    for i in range(m["num_layers"]):
        x = _encoder_block(x, m, i + 1)
    x = layers.LayerNormalization(epsilon=1e-6, name="encoder_norm")(x)
    return x[:, 0]                                                                 # [CLS] token


def _pretrained_vit(x, m):
    import keras_hub  # pip install keras-hub

    backbone = keras_hub.models.ViTBackbone.from_preset(m.get("preset", "vit_base_patch16_224_imagenet"))
    sequence = backbone(x)
    return sequence[:, 0]                                                          # [CLS] token


def build(cfg: dict) -> keras.Model:
    m = cfg["model"]
    size = cfg["data"]["image_size"]
    num_classes = len(cfg["data"]["classes"])

    inputs = keras.Input((size, size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(inputs)       # 0-255 -> [-1, 1]
    x = _pretrained_vit(x, m) if m.get("pretrained", True) else _scratch_vit(x, m)
    outputs = classification_head(x, num_classes, m.get("dropout", 0.1))
    return keras.Model(inputs, outputs, name="vit")
