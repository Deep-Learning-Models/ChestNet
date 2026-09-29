"""Plots for training curves, ROC curves and confusion matrices."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def plot_history(history_csv, out_path, title):
    h = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["loss", "auc"]):
        if metric in h:
            ax.plot(h["epoch"] + 1, h[metric], label="train")
        if f"val_{metric}" in h:
            ax.plot(h["epoch"] + 1, h[f"val_{metric}"], label="validation")
        ax.set_title(f"{title} – {metric.upper()}")
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(y_true, y_prob, classes, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, c in enumerate(classes):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=c)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{title} – ROC curves (test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(cms, classes, out_path, title):
    cols = 4
    rows = int(np.ceil(len(classes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3 * rows))
    for ax, cm, c in zip(np.ravel(axes), cms, classes):
        ax.imshow(cm, cmap="Blues")
        for (r, k), v in np.ndenumerate(cm):
            ax.text(k, r, int(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["neg", "pos"])
        ax.set_yticks([0, 1], ["neg", "pos"])
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    for ax in np.ravel(axes)[len(classes):]:
        ax.axis("off")
    fig.suptitle(f"{title} – confusion matrix per class (test)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
