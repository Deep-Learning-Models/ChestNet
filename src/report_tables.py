"""Print report-ready Markdown tables from the files written by train.py and evaluate.py.

Run AFTER training and test evaluation of every model:
    python -m src.report_tables            (prints to screen and saves results/report_tables.md)

Reads, for each model:
    results/<model>/training_summary.json   (params, time per epoch, best epoch, best val AUC, hardware)
    results/<model>/test_metrics.json       (macro metrics, inference ms per image)
    results/<model>/test_per_class_metrics.csv
    results/<model>/history.csv             (for the overfitting table)
"""
import json
from pathlib import Path

import pandas as pd

RESULTS = Path("results")
MODELS = [("custom_cnn", "Custom CNN"), ("resnet50", "ResNet50"),
          ("efficientnet", "EfficientNet-B0"), ("vit", "ViT-B/16")]


def _load(model):
    d = RESULTS / model
    out = {}
    for key, fname in [("train", "training_summary.json"), ("test", "test_metrics.json")]:
        p = d / fname
        out[key] = json.load(open(p, encoding="utf-8")) if p.exists() else None
    p = d / "test_per_class_metrics.csv"
    out["per_class"] = pd.read_csv(p) if p.exists() else None
    p = d / "history.csv"
    out["history"] = pd.read_csv(p) if p.exists() else None
    return out


def _f(x, nd=3):
    return "–" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def main():
    data = {m: _load(m) for m, _ in MODELS}
    lines = []

    # Table A: overall test performance
    lines += ["### Test-set performance (threshold 0.5)", "",
              "| Model | Macro ROC-AUC | Macro F1 | Macro Precision | Macro Recall | Micro F1 | Label accuracy | Subset (exact-match) accuracy |",
              "|---|---|---|---|---|---|---|---|"]
    for m, label in MODELS:
        t = data[m]["test"] or {}
        lines.append(f"| {label} | {_f(t.get('macro_roc_auc'))} | {_f(t.get('macro_f1'))} | "
                     f"{_f(t.get('macro_precision'))} | {_f(t.get('macro_recall'))} | {_f(t.get('micro_f1'))} | "
                     f"{_f(t.get('label_accuracy'))} | {_f(t.get('subset_accuracy'))} |")

    # Table B: per-class AUC and F1
    for metric in ["roc_auc", "f1", "recall", "precision"]:
        lines += ["", f"### Per-class test {metric}", "",
                  "| Class | " + " | ".join(l for _, l in MODELS) + " |",
                  "|---|" + "---|" * len(MODELS)]
        classes = next((d["per_class"]["class"].tolist() for d in data.values() if d["per_class"] is not None), [])
        for c in classes:
            row = []
            for m, _ in MODELS:
                pc = data[m]["per_class"]
                row.append(_f(pc.set_index("class").loc[c, metric]) if pc is not None else "–")
            lines.append(f"| {c} | " + " | ".join(row) + " |")

    # Table C: cost
    lines += ["", "### Computational cost", "",
              "| Model | Total params | Trainable params | Mean time / epoch (s) | Epochs run | Best epoch | Total train time (min) | Inference (ms / image) | Hardware |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m, label in MODELS:
        tr, te = data[m]["train"] or {}, data[m]["test"] or {}
        hw = ", ".join((tr.get("hardware") or {}).get("gpus", [])) or "–"
        lines.append(f"| {label} | {tr.get('total_params', '–'):,} | {tr.get('trainable_params', '–'):,} | "
                     f"{_f(tr.get('mean_epoch_time_s'), 1)} | {tr.get('epochs_run', '–')} | {tr.get('best_epoch', '–')} | "
                     f"{_f(tr.get('total_train_time_min'), 1)} | {_f(te.get('inference_ms_per_image'), 2)} | {hw} |"
                     if tr else f"| {label} | – | – | – | – | – | – | – | – |")

    # Table D: generalisation / overfitting
    lines += ["", "### Generalisation and overfitting", "",
              "| Model | Best val AUC | Test AUC | Val–test gap | Train AUC at best epoch | Train–val AUC gap | Final train loss | Final val loss |",
              "|---|---|---|---|---|---|---|---|"]
    for m, label in MODELS:
        tr, te, h = data[m]["train"] or {}, data[m]["test"] or {}, data[m]["history"]
        bv, ta = tr.get("best_val_auc"), te.get("macro_roc_auc")
        gap = bv - ta if bv is not None and ta is not None else None
        if h is not None and tr.get("best_epoch"):
            r = h.iloc[tr["best_epoch"] - 1]
            tr_auc, tv_gap = r.get("auc"), r.get("auc") - r.get("val_auc")
            fl, fvl = h["loss"].iloc[-1], h["val_loss"].iloc[-1]
        else:
            tr_auc = tv_gap = fl = fvl = None
        lines.append(f"| {label} | {_f(bv)} | {_f(ta)} | {_f(gap)} | {_f(tr_auc)} | {_f(tv_gap)} | {_f(fl)} | {_f(fvl)} |")

    text = "\n".join(lines)
    print(text)
    (RESULTS / "report_tables.md").write_text(text, encoding="utf-8")
    print(f"\n[report_tables] saved {RESULTS / 'report_tables.md'}")


if __name__ == "__main__":
    main()
