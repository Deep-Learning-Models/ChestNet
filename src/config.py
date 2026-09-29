"""Load a model config on top of the shared base config (configs/base.yaml).

Step 7 – shared training rules
------------------------------
Some settings MUST be identical for all four models, otherwise the comparison is not
fair (e.g. one model seeing bigger images or more epochs). These are LOCKED: a model
config (configs/resnet50.yaml, ...) is not allowed to change them, and load_config()
stops with a clear error if it tries.

Settings that legitimately depend on the architecture (learning rate, weight decay,
dropout, batch size for GPU memory, the model block itself) may be overridden – each
override is recorded so it can be reported and justified.
"""
import copy
from pathlib import Path

import yaml

BASE_CONFIG = Path("configs/base.yaml")

# Dotted keys that every model must share (fair comparison).
LOCKED_KEYS = [
    "seed",
    "data.csv_path", "data.image_dir", "data.splits_dir", "data.classes",
    "data.max_images_per_class", "data.val_size", "data.test_size", "data.image_size",
    "augmentation",
    "training.epochs", "training.optimizer", "training.lr_schedule",
    "training.early_stopping_patience", "training.monitor",
    "training.use_class_weights", "training.loss", "training.threshold",
]

# Dotted keys a model may change, with the reason to cite in the report.
ALLOWED_OVERRIDES = {
    "model": "architecture-specific settings",
    "training.learning_rate": "pretrained models need a small LR (keep ImageNet features); from-scratch models a larger one",
    "training.weight_decay": "transformers are usually regularised more strongly",
    "data.batch_size": "large models (ViT-B/16) need a smaller batch to fit in GPU memory",
}


class ProtocolViolation(ValueError):
    """Raised when a model config tries to change a locked shared setting."""


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _is_under(key: str, roots) -> str:
    for r in roots:
        if key == r or key.startswith(r + "."):
            return r
    return ""


def check_protocol(base: dict, model_cfg: dict, source: str = "model config") -> list:
    """Return the list of allowed overrides; raise ProtocolViolation on a locked key."""
    overrides = []
    for key, value in _flatten(model_cfg).items():
        locked = _is_under(key, LOCKED_KEYS)
        if locked:
            base_value = _flatten(base).get(key, base.get(key))
            if value != base_value:
                raise ProtocolViolation(
                    f"{source} changes '{key}' ({base_value!r} -> {value!r}), but '{locked}' is a "
                    f"shared rule for ALL models (see configs/base.yaml). Change it in base.yaml "
                    f"instead so every model uses the same value.")
            continue
        allowed = _is_under(key, ALLOWED_OVERRIDES)
        if not allowed:
            raise ProtocolViolation(
                f"{source} sets '{key}', which is not in ALLOWED_OVERRIDES (src/config.py). "
                f"Add it to base.yaml or to ALLOWED_OVERRIDES with a justification.")
        if key.startswith("model"):
            continue
        overrides.append({"key": key, "value": value, "base": _flatten(base).get(key),
                          "reason": ALLOWED_OVERRIDES[allowed]})
    return overrides


def load_config(model_config_path: str) -> dict:
    """Return base.yaml merged with the given model config (after the fairness check)."""
    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    if Path(model_config_path).resolve() == BASE_CONFIG.resolve():
        cfg = copy.deepcopy(base)
        cfg["protocol_overrides"] = []
        return cfg
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}
    overrides = check_protocol(base, model_cfg, source=str(model_config_path))
    cfg = _deep_merge(base, model_cfg)
    cfg["protocol_overrides"] = overrides   # saved with every run for the report
    return cfg


def save_config(cfg: dict, path: Path) -> None:
    """Save the exact config used for a run (reproducibility)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
