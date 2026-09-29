"""Evaluation metrics for multi-label chest X-ray classification."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, multilabel_confusion_matrix,
                             precision_score, recall_score, roc_auc_score)


def _safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, classes, threshold: float = 0.5):
    """Return (summary dict, per-class DataFrame, multilabel confusion matrices)."""
    y_pred = (y_prob >= threshold).astype(int)

    per_class = pd.DataFrame({
        "class": classes,
        "support": y_true.sum(axis=0).astype(int),
        "precision": precision_score(y_true, y_pred, average=None, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=None, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=None, zero_division=0),
        "roc_auc": [_safe_auc(y_true[:, i], y_prob[:, i]) for i in range(len(classes))],
    })

    summary = {
        "threshold": threshold,
        "macro_precision": float(per_class["precision"].mean()),
        "macro_recall": float(per_class["recall"].mean()),
        "macro_f1": float(per_class["f1"].mean()),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_roc_auc": float(np.nanmean(per_class["roc_auc"])),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),   # exact match (strict)
        "label_accuracy": float((y_true == y_pred).mean()),          # per-label (context only)
    }
    return summary, per_class, multilabel_confusion_matrix(y_true, y_pred)
