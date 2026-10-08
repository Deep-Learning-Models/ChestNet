"""Tune one decision threshold per disease on VALIDATION, then apply it to TEST.

Why? 🎚️  A sigmoid output of 0.5 is an arbitrary cut-off. For a rare disease such as
Pneumonia the model may rarely reach 0.5 even when it ranks the sick patients highest
(good AUC, but recall ~0). Picking the threshold that maximises F1 on the VALIDATION set
fixes this without ever looking at the test labels (no leakage).

Run after evaluate.py has been run on BOTH splits:
    python -m src.evaluate --config configs/efficientnet.yaml --split val
    python -m src.evaluate --config configs/efficientnet.yaml --split test
    python -m src.tune_thresholds --config configs/efficientnet.yaml

Outputs in results/<run>/:
    thresholds.csv                   threshold per class + validation F1 at 0.5 and at the tuned value
    test_metrics_tuned.json          test metrics with the tuned thresholds
    test_per_class_metrics_tuned.csv
The fixed-0.5 results stay the official comparison numbers; the tuned ones are an extra.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import load_config, run_name
from src.preprocessing.splits import load_splits
from src.utils.metrics import compute_metrics

GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def best_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """For each class, the threshold in GRID with the highest F1 (ties -> the one closest to 0.5)."""
    out = []
    for c in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, c], y_prob[:, c] >= t, zero_division=0) for t in GRID]
        best = max(range(len(GRID)), key=lambda i: (scores[i], -abs(GRID[i] - 0.5)))
        out.append(float(GRID[best]))
    return np.array(out)


def main():
    parser = argparse.ArgumentParser(description="Per-class threshold tuning on the validation split")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = run_name(cfg)
    classes = cfg["data"]["classes"]
    d = Path(cfg["output"]["results_dir"]) / name
    for split in ("val", "test"):
        if not (d / f"{split}_probabilities.npy").exists():
            raise SystemExit(f"Missing {d}/{split}_probabilities.npy – run "
                             f"`python -m src.evaluate --config {args.config} --split {split}` first.")

    _, val_df, test_df = load_splits(cfg)
    yv, pv = val_df[classes].to_numpy(), np.load(d / "val_probabilities.npy")
    yt, pt = test_df[classes].to_numpy(), np.load(d / "test_probabilities.npy")

    th = best_thresholds(yv, pv)                                   # chosen on VALIDATION only
    table = pd.DataFrame({
        "class": classes,
        "threshold": th,
        "val_f1_at_0.5": [f1_score(yv[:, i], pv[:, i] >= 0.5, zero_division=0) for i in range(len(classes))],
        "val_f1_tuned": [f1_score(yv[:, i], pv[:, i] >= th[i], zero_division=0) for i in range(len(classes))],
    })
    table.to_csv(d / "thresholds.csv", index=False)

    summary, per_class, _ = compute_metrics(yt, pt, classes, th)   # applied ONCE to test
    summary["threshold"] = {c: float(t) for c, t in zip(classes, th)}
    summary.update({"model": name, "split": "test", "thresholds_tuned_on": "val"})
    with open(d / "test_metrics_tuned.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    per_class.to_csv(d / "test_per_class_metrics_tuned.csv", index=False)

    base = json.load(open(d / "test_metrics.json")) if (d / "test_metrics.json").exists() else {}
    print(table.round(3).to_string(index=False))
    print(f"\n[thresholds] test macro F1: {base.get('macro_f1', float('nan')):.4f} at 0.5  ->  "
          f"{summary['macro_f1']:.4f} with validation-tuned thresholds. Saved to {d}")


if __name__ == "__main__":
    main()
