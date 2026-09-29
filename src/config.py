"""Load a model config on top of the shared base config (configs/base.yaml)."""
import copy
from pathlib import Path

import yaml

BASE_CONFIG = Path("configs/base.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(model_config_path: str) -> dict:
    """Return base.yaml merged with the given model config."""
    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    return _deep_merge(base, model_cfg)


def save_config(cfg: dict, path: Path) -> None:
    """Save the exact config used for a run (reproducibility)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
