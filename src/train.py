"""Train one ChestNet model.

Usage:
    python -m src.train --config configs/resnet50.yaml
    python -m src.train --config configs/custom_cnn.yaml --epochs 5      # quick test run
"""
import argparse
import json
import platform
import time
from pathlib import Path

import keras
import tensorflow as tf

from src.config import load_config, save_config
from src.models import build_model
from src.preprocessing.dataset import compute_pos_weights, make_dataset
from src.preprocessing.splits import load_splits
from src.utils.losses import weighted_bce
from src.utils.plots import plot_history
from src.utils.seed import set_seed


class EpochTimer(keras.callbacks.Callback):
    """Record wall-clock time per epoch (for the computational-cost comparison)."""

    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self._start)
        if logs is not None:
            logs["epoch_time_s"] = self.times[-1]


def build_optimizer(cfg: dict, steps_per_epoch: int):
    t = cfg["training"]
    lr = t["learning_rate"]
    if t.get("lr_schedule") == "cosine":
        lr = keras.optimizers.schedules.CosineDecay(lr, decay_steps=max(1, t["epochs"] * steps_per_epoch))
    if t.get("optimizer", "adamw").lower() == "adamw":
        return keras.optimizers.AdamW(learning_rate=lr, weight_decay=t.get("weight_decay", 1e-4))
    return keras.optimizers.Adam(learning_rate=lr)


def main():
    parser = argparse.ArgumentParser(description="Train a ChestNet model")
    parser.add_argument("--config", required=True, help="e.g. configs/resnet50.yaml")
    parser.add_argument("--epochs", type=int, help="override the number of epochs")
    parser.add_argument("--deterministic", action="store_true", help="fully deterministic GPU ops (slower)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    set_seed(cfg["seed"], deterministic_ops=args.deterministic)

    name = cfg["model"]["name"]
    classes = cfg["data"]["classes"]
    results_dir = Path(cfg["output"]["results_dir"]) / name
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, results_dir / "config_used.yaml")

    # ---- data (same patient-level splits for every model) ----
    train_df, val_df, _ = load_splits(cfg)
    train_ds = make_dataset(train_df, cfg, training=True)
    val_ds = make_dataset(val_df, cfg, training=False)
    steps_per_epoch = len(train_ds)
    print(f"[train] {name}: {len(train_df)} train / {len(val_df)} val images")

    # ---- model ----
    model = build_model(cfg)
    pos_weights = compute_pos_weights(train_df, classes) if cfg["training"].get("use_class_weights") else [1.0] * len(classes)
    model.compile(
        optimizer=build_optimizer(cfg, steps_per_epoch),
        loss=weighted_bce(pos_weights),
        metrics=[
            keras.metrics.AUC(multi_label=True, num_labels=len(classes), name="auc"),
            keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=cfg["training"]["threshold"]),
        ],
    )
    model.summary(print_fn=lambda s, **_: None)
    total_params = model.count_params()
    trainable_params = int(sum(keras.ops.size(w) for w in model.trainable_weights))
    print(f"[train] parameters: {total_params:,} total / {trainable_params:,} trainable")

    ckpt_path = ckpt_dir / f"{name}_best.weights.h5"
    timer = EpochTimer()
    callbacks = [
        timer,
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_auc", mode="max",
                                        save_best_only=True, save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", restore_best_weights=True,
                                      patience=cfg["training"]["early_stopping_patience"], verbose=1),
        keras.callbacks.CSVLogger(results_dir / "history.csv"),
    ]

    start = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg["training"]["epochs"], callbacks=callbacks)
    total_time = time.perf_counter() - start

    # ---- save training summary (for the cost comparison in the report) ----
    val_auc = history.history.get("val_auc", [float("nan")])
    summary = {
        "model": name,
        "config": args.config,
        "seed": cfg["seed"],
        "total_params": int(total_params),
        "trainable_params": trainable_params,
        "epochs_run": len(timer.times),
        "best_epoch": int(max(range(len(val_auc)), key=lambda i: val_auc[i])) + 1,
        "best_val_auc": float(max(val_auc)),
        "mean_epoch_time_s": float(sum(timer.times) / max(1, len(timer.times))),
        "total_train_time_min": total_time / 60,
        "hardware": {
            "gpus": [d.name for d in tf.config.list_physical_devices("GPU")] or ["CPU only"],
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
        },
        "checkpoint": str(ckpt_path),
    }
    with open(results_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    plot_history(results_dir / "history.csv", results_dir / "training_curves.png", name)
    print(f"[train] done – best val AUC {summary['best_val_auc']:.4f}. Results in {results_dir}")


if __name__ == "__main__":
    main()
