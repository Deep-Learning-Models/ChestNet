"""Evaluate a trained model on the held-out TEST split (run once per model, at the end).

Usage:
    python -m src.evaluate --config configs/resnet50.yaml
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.models import build_model
from src.preprocessing.dataset import make_dataset
from src.preprocessing.splits import load_splits
from src.utils.metrics import compute_metrics
from src.utils.plots import plot_confusion_matrices, plot_roc_curves
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate a ChestNet model on the test set")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="defaults to saved_models/<model>_best.weights.h5")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    name = cfg["model"]["name"]
    classes = cfg["data"]["classes"]
    results_dir = Path(cfg["output"]["results_dir"]) / name
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.checkpoint or Path(cfg["output"]["checkpoint_dir"]) / f"{name}_best.weights.h5"

    train_df, val_df, test_df = load_splits(cfg)
    df = test_df if args.split == "test" else val_df
    ds = make_dataset(df, cfg, training=False)

    model = build_model(cfg)
    model.load_weights(ckpt)

    # Warm-up once, then time inference over the whole split.
    model.predict(ds.take(1), verbose=0)
    start = time.perf_counter()
    y_prob = model.predict(ds, verbose=1)
    ms_per_image = 1000 * (time.perf_counter() - start) / len(df)
    y_true = df[classes].to_numpy()

    summary, per_class, cms = compute_metrics(y_true, y_prob, classes, cfg["training"]["threshold"])
    summary.update({"model": name, "split": args.split, "images": len(df),
                    "params": int(model.count_params()), "inference_ms_per_image": ms_per_image})

    with open(results_dir / f"{args.split}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    per_class.to_csv(results_dir / f"{args.split}_per_class_metrics.csv", index=False)
    np.save(results_dir / f"{args.split}_probabilities.npy", y_prob)
    plot_roc_curves(y_true, y_prob, classes, results_dir / f"{args.split}_roc_curves.png", name)
    plot_confusion_matrices(cms, classes, results_dir / f"{args.split}_confusion_matrices.png", name)

    # One shared comparison table for all four models.
    if args.split == "test":
        train_summary = results_dir / "training_summary.json"
        extra = json.load(open(train_summary)) if train_summary.exists() else {}
        row = {
            "model": name,
            "macro_f1": summary["macro_f1"],
            "macro_roc_auc": summary["macro_roc_auc"],
            "macro_precision": summary["macro_precision"],
            "macro_recall": summary["macro_recall"],
            "params": summary["params"],
            "mean_epoch_time_s": extra.get("mean_epoch_time_s"),
            "best_val_auc": extra.get("best_val_auc"),
            "val_test_auc_gap": (extra["best_val_auc"] - summary["macro_roc_auc"]) if extra else None,
            "inference_ms_per_image": ms_per_image,
        }
        comp_path = Path(cfg["output"]["results_dir"]) / "comparison.csv"
        comp = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()
        comp = pd.concat([comp[comp.get("model", pd.Series(dtype=str)) != name] if len(comp) else comp,
                          pd.DataFrame([row])], ignore_index=True)
        comp.to_csv(comp_path, index=False)

    print(per_class.round(3).to_string(index=False))
    print(f"\n[evaluate] macro F1 {summary['macro_f1']:.4f} | macro ROC-AUC {summary['macro_roc_auc']:.4f} "
          f"| {ms_per_image:.2f} ms/image. Saved to {results_dir}")


if __name__ == "__main__":
    main()
