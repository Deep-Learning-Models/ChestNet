"""Model registry: build any of the four models by name."""
from src.models import custom_cnn, efficientnet, resnet50, vit

MODELS = {
    "custom_cnn": custom_cnn,
    "resnet50": resnet50,
    "efficientnet": efficientnet,
    "vit": vit,
}


def build_model(cfg: dict):
    name = cfg["model"]["name"]
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS)}")
    return MODELS[name].build(cfg)


def gradcam_layer(cfg: dict):
    return MODELS[cfg["model"]["name"]].GRADCAM_LAYER
