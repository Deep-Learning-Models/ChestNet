"""Step 7: verify the shared training rules BEFORE any model is trained.

    python -m src.check_pipeline

It checks (and saves evidence for the report in results/protocol/):
  1. Every model config obeys the shared protocol (no locked setting is changed).
  2. The patient-level splits exist and share NO patient and NO image.
  3. Every class has positives in train, validation and test.
  4. The tf.data pipeline returns the right shapes, dtype and pixel range (0-255).
  5. Augmentation changes TRAINING images only; validation images are identical every pass.
  6. Class weights are computed from the TRAINING split only.
  7. Data-loading speed and hardware are logged.

Outputs:
  results/protocol/experimental_protocol.json   the full shared protocol + per-model overrides
  results/protocol/experimental_protocol.md     the same as a table to paste into Report Section 5
  results/protocol/class_weights.csv            positive-class weights (train split)
  results/protocol/augmentation_preview.png     original vs augmented training images
"""
import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import ALLOWED_OVERRIDES, BASE_CONFIG, LOCKED_KEYS, _flatten, load_config
from src.preprocessing.dataset import build_augmenter, compute_pos_weights, make_dataset, _load_image
from src.preprocessing.splits import leakage_report, load_splits
from src.utils.seed import set_seed

MODEL_CONFIGS = ["configs/custom_cnn.yaml", "configs/resnet50.yaml",
                 "configs/efficientnet.yaml", "configs/vit.yaml"]


def _ok(msg):
    print(f"  ✅ {msg}")


def check_protocol(model_configs) -> dict:
    print("\n[1] Shared protocol across models")
    per_model = {}
    for path in model_configs:
        cfg = load_config(path)            # raises ProtocolViolation if a locked key changed
        per_model[cfg["model"]["name"]] = cfg
        ov = ", ".join(f"{o['key']}={o['value']}" for o in cfg["protocol_overrides"]) or "none"
        _ok(f"{path}: locked settings unchanged (overrides: {ov})")
    return per_model


def check_splits(cfg) -> tuple:
    print("\n[2-3] Patient-level splits")
    train, val, test = load_splits(cfg)
    rep = leakage_report(train, val, test)
    assert rep["patient_overlap_total"] == 0, f"Patient leakage: {rep}"
    assert rep["image_overlap_total"] == 0, f"Image leakage: {rep}"
    _ok(f"0 shared patients / 0 shared images across train ({len(train)}), val ({len(val)}), test ({len(test)})")
    classes = cfg["data"]["classes"]
    for name, part in [("train", train), ("val", val), ("test", test)]:
        missing = [c for c in classes if part[c].sum() == 0]
        assert not missing, f"{name} split has no positives for {missing}"
    _ok("every class has positives in every split")
    return train, val, test, rep


def check_pipeline(cfg, train, val) -> dict:
    print("\n[4-5] tf.data pipeline")
    size, n_cls = cfg["data"]["image_size"], len(cfg["data"]["classes"])
    # Partial reads (one batch) would leave a half-written cache, so the shape checks run uncached.
    nocache = {**cfg, "data": {**cfg["data"], "cache": "none"}}
    train_ds = make_dataset(train, nocache, training=True, split="train")
    val_ds = make_dataset(val, nocache, training=False, split="val")

    x, y = next(iter(train_ds))
    assert x.shape[1:] == (size, size, 3) and x.dtype == tf.float32, f"bad image batch {x.shape} {x.dtype}"
    assert y.shape[1] == n_cls and set(np.unique(y.numpy())) <= {0.0, 1.0}, "labels must be multi-hot 0/1"
    xmin, xmax = float(tf.reduce_min(x)), float(tf.reduce_max(x))
    assert 0.0 <= xmin and xmax <= 255.0, f"pixel range {xmin}-{xmax} (expected 0-255)"
    _ok(f"train batch {tuple(x.shape)} float32, pixels {xmin:.0f}-{xmax:.0f}, labels multi-hot {tuple(y.shape)}")

    v1 = next(iter(val_ds))[0].numpy()
    v2 = next(iter(val_ds))[0].numpy()
    assert np.array_equal(v1, v2), "validation images changed between passes (augmentation leak?)"
    _ok("validation images are identical on every pass (no augmentation, fixed order)")

    # Timing: first pass fills the cache, second pass reads from it.
    val_ds = make_dataset(val, cfg, training=False, split="val")
    timings = {}
    for label in ("first_pass_s", "cached_pass_s"):
        t = time.perf_counter()
        for _ in val_ds:
            pass
        timings[label] = round(time.perf_counter() - t, 2)
    _ok(f"validation pass: {timings['first_pass_s']} s (fills cache) → {timings['cached_pass_s']} s (cached)")
    return timings


def augmentation_preview(cfg, train, out: Path, n: int = 4):
    aug = build_augmenter(cfg.get("augmentation", {}), cfg["seed"])
    rows = train.sample(n, random_state=cfg["seed"])
    fig, axes = plt.subplots(n, 4, figsize=(9, 2.4 * n))
    for r, (_, row) in enumerate(rows.iterrows()):
        img = _load_image(row["path"], cfg["data"]["image_size"])
        axes[r, 0].imshow(img.numpy()[..., 0], cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_title("original" if r == 0 else "", fontsize=9)
        for k in range(1, 4):
            a = tf.clip_by_value(aug(img[None], training=True)[0], 0, 255)
            axes[r, k].imshow(a.numpy()[..., 0], cmap="gray", vmin=0, vmax=255)
            axes[r, k].set_title(f"augmented {k}" if r == 0 else "", fontsize=9)
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle("Training-only augmentation (validation and test images are never changed)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "augmentation_preview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _ok("saved augmentation_preview.png")


def protocol_table(base_cfg, per_model) -> str:
    flat = _flatten({k: v for k, v in base_cfg.items() if k not in ("output", "protocol_overrides")})
    locked = {k: v for k, v in flat.items() if any(k == l or k.startswith(l + ".") for l in LOCKED_KEYS)}
    lines = ["# ChestNet – shared experimental protocol", "",
             "## 🔒 Identical for all four models", "", "| Setting | Value |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in locked.items()]
    lines += ["", "## ✏️ Architecture-specific settings (allowed to differ)", "",
              "| Setting | " + " | ".join(per_model) + " | Why it may differ |",
              "|---|" + "---|" * (len(per_model) + 1)]
    keys = [k for k in ALLOWED_OVERRIDES if k != "model"]
    for k in keys:
        vals = [str(_flatten(c).get(k)) for c in per_model.values()]
        lines.append(f"| `{k}` | " + " | ".join(vals) + f" | {ALLOWED_OVERRIDES[k]} |")
    lines += ["", "Normalisation is applied inside each model (Custom CNN: /255, ResNet50: caffe BGR "
              "mean subtraction, EfficientNet: built-in, ViT: [-1, 1]); the input pipeline is identical."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Verify the shared ChestNet pipeline (Step 7)")
    parser.add_argument("--config", default=str(BASE_CONFIG))
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out = Path(cfg["output"]["results_dir"]) / "protocol"
    out.mkdir(parents=True, exist_ok=True)

    per_model = check_protocol([p for p in MODEL_CONFIGS if Path(p).exists()])
    train, val, test, leak = check_splits(cfg)
    timings = {} if args.skip_timing else check_pipeline(cfg, train, val)

    print("\n[6] Class weights (train split only)")
    classes = cfg["data"]["classes"]
    w = compute_pos_weights(train, classes, cfg["training"].get("max_pos_weight"))
    weights = pd.DataFrame({"class": classes, "train_positives": train[classes].sum().values.astype(int),
                            "pos_weight": np.round(w, 3)})
    weights.to_csv(out / "class_weights.csv", index=False)
    print(weights.to_string(index=False))

    print("\n[7] Evidence files")
    augmentation_preview(cfg, train, out)
    protocol = {
        "shared": {k: v for k, v in cfg.items() if k not in ("model", "protocol_overrides", "output")},
        "per_model_overrides": {n: c["protocol_overrides"] for n, c in per_model.items()},
        "leakage_report": leak,
        "class_weights": dict(zip(classes, map(float, np.round(w, 3)))),
        "data_loading_timing": timings,
        "hardware": {"gpus": [d.name for d in tf.config.list_physical_devices("GPU")] or ["CPU only"],
                     "platform": platform.platform(), "python": platform.python_version(),
                     "tensorflow": tf.__version__},
    }
    with open(out / "experimental_protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2, default=str)
    (out / "experimental_protocol.md").write_text(protocol_table(cfg, per_model), encoding="utf-8")
    _ok(f"saved experimental_protocol.json / .md and class_weights.csv in {out}")
    print("\n🎉 All shared training rules verified – the pipeline is ready for model training (Step 8).")


if __name__ == "__main__":
    main()
