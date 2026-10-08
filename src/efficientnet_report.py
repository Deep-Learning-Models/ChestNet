"""EfficientNet B0 vs B3 report figures (Member 3).

Run after both models are trained and evaluated on test:
    python -m src.efficientnet_report
    python -m src.efficientnet_report --runs efficientnet efficientnet_b3 efficientnet_b3_300 --num-images 6

Creates in results/efficientnet_report/:
    scoreboard.png        🏁 test macro AUC vs parameters (bubble size = inference time) for EVERY
                             model in results/comparison.csv -> "accuracy per parameter" at a glance
    scoreboard.csv        the same numbers + AUC per million parameters
    b0_vs_b3_curves.png   📈 validation AUC / loss per epoch, all EfficientNet runs on one plot
    per_class_auc.png     🩺 which diseases gain from the bigger B3
    gradcam_b0_vs_b3.png  🔥 the SAME test X-rays explained by each run, for the TRUE disease
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
CONFIGS = {"efficientnet": "configs/efficientnet.yaml",
           "efficientnet_b3": "configs/efficientnet_b3.yaml",
           "efficientnet_b3_300": "configs/efficientnet_b3_300.yaml"}
LABELS = {"efficientnet": "EffNet-B0 (224)", "efficientnet_b3": "EffNet-B3 (224)",
          "efficientnet_b3_300": "EffNet-B3 (300, ablation)", "custom_cnn": "Custom CNN",
          "resnet50": "ResNet50", "vit": "ViT-B/16"}


def scoreboard(out: Path):
    comp_path = RESULTS / "comparison.csv"
    if not comp_path.exists():
        print("[report] no results/comparison.csv yet – run src.evaluate on the test split first")
        return
    comp = pd.read_csv(comp_path)
    comp["params_M"] = comp["params"] / 1e6
    comp["auc_per_M_params"] = comp["macro_roc_auc"] / comp["params_M"]
    comp.sort_values("macro_roc_auc", ascending=False).to_csv(out / "scoreboard.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ms = comp["inference_ms_per_image"].fillna(comp["inference_ms_per_image"].median()).fillna(1.0)
    sizes = 120 + 700 * (ms - ms.min()) / max(float(ms.max() - ms.min()), 1e-9)   # relative bubble size
    for (_, r), size in zip(comp.iterrows(), sizes):
        eff = str(r["model"]).startswith("efficientnet")
        ax.scatter(r["params_M"], r["macro_roc_auc"], s=size, alpha=0.7,
                   color="#2a9d8f" if eff else "#8d99ae", edgecolor="black", zorder=3)
        ax.annotate(f"{LABELS.get(r['model'], r['model'])}\n{r['inference_ms_per_image']:.1f} ms",
                    (r["params_M"], r["macro_roc_auc"]), textcoords="offset points", xytext=(12, 4), fontsize=8)
    ax.margins(x=0.35, y=0.25)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if v in (2, 5, 20, 50) else ""))
    ax.set_xlabel("parameters (millions, log scale)  ← smaller is cheaper")
    ax.set_ylabel("test macro ROC-AUC  ↑ better")
    ax.set_title("ChestNet efficiency scoreboard (bubble size = inference ms / image)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "scoreboard.png", dpi=150)
    plt.close(fig)
    print(comp[["model", "macro_roc_auc", "macro_f1", "params_M", "auc_per_M_params",
                "inference_ms_per_image"]].round(4).to_string(index=False))


def curves(runs, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run in runs:
        h_path = RESULTS / run / "history.csv"
        if not h_path.exists():
            continue
        h = pd.read_csv(h_path)
        for ax, metric in zip(axes, ["val_auc", "val_loss"]):
            line, = ax.plot(h["epoch"] + 1, h[metric], marker="o", ms=3, label=f"{LABELS.get(run, run)} val")
            ax.plot(h["epoch"] + 1, h[metric.replace("val_", "")], ls="--", color=line.get_color(), alpha=0.6,
                    label=f"{LABELS.get(run, run)} train")
    for ax, t in zip(axes, ["AUC (higher = better)", "loss (lower = better)"]):
        ax.set_title(t); ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("EfficientNet runs: solid = validation, dashed = training (a growing gap = overfitting)")
    fig.tight_layout()
    fig.savefig(out / "b0_vs_b3_curves.png", dpi=150)
    plt.close(fig)


def per_class(runs, out: Path):
    tables = {}
    for run in runs:
        p = RESULTS / run / "test_per_class_metrics.csv"
        if p.exists():
            tables[LABELS.get(run, run)] = pd.read_csv(p).set_index("class")["roc_auc"]
    if not tables:
        return
    df = pd.DataFrame(tables)
    ax = df.plot.barh(figsize=(8, 5), width=0.8)
    ax.axvline(0.5, color="red", ls=":", label="random guess")
    ax.set_xlim(0.4, 1.0); ax.set_xlabel("test ROC-AUC"); ax.set_title("Per-disease test AUC")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")
    plt.tight_layout(); plt.savefig(out / "per_class_auc.png", dpi=150); plt.close()
    df.round(3).to_csv(out / "per_class_auc.csv")


def gradcam_compare(runs, num_images: int, out: Path):
    from src.config import load_config
    from src.models import build_model, gradcam_layer
    from src.preprocessing.dataset import _load_image
    from src.preprocessing.splits import load_splits
    from src.utils.gradcam import gradcam_heatmap

    loaded = []
    for run in runs:
        cfg = load_config(CONFIGS[run])
        ckpt = Path(cfg["output"]["checkpoint_dir"]) / f"{run}_best.weights.h5"
        if not ckpt.exists():
            print(f"[report] skip Grad-CAM for {run}: {ckpt} not found")
            continue
        model = build_model(cfg)
        model.load_weights(ckpt)
        loaded.append((run, cfg, model, gradcam_layer(cfg)))
    if not loaded:
        return

    cfg0 = loaded[0][1]
    classes = cfg0["data"]["classes"]
    diseases = [c for c in classes if c != "No Finding"]
    _, _, test_df = load_splits(cfg0)
    sick = test_df[test_df[diseases].sum(axis=1) > 0]
    # One example per disease where possible, so the figure covers different findings.
    picks = []
    for c in diseases:
        rows = sick[(sick[c] == 1) & (~sick.index.isin(picks))]
        if len(rows):
            picks.append(rows.sample(1, random_state=cfg0["seed"]).index[0])
    sample = sick.loc[picks[:num_images]]

    fig, axes = plt.subplots(len(sample), len(loaded) + 1, figsize=(3.3 * (len(loaded) + 1), 3.4 * len(sample)))
    axes = np.atleast_2d(axes)
    for row, (_, r) in zip(axes, sample.iterrows()):
        true = [c for c in diseases if r[c] == 1]
        base_img = _load_image(r["path"], cfg0["data"]["image_size"]).numpy()
        row[0].imshow(base_img.astype("uint8")); row[0].set_title("True: " + ", ".join(true), fontsize=8)
        for ax, (run, cfg, model, layer) in zip(row[1:], loaded):
            img = _load_image(r["path"], cfg["data"]["image_size"]).numpy()   # 300 px for the ablation
            probs = model.predict(img[None], verbose=0)[0]
            c = max(true, key=lambda k: probs[classes.index(k)])            # explain the TRUE disease
            heat = gradcam_heatmap(model, img, layer, classes.index(c))
            ax.imshow(img.astype("uint8")); ax.imshow(heat, cmap="jet", alpha=0.4)
            ax.set_title(f"{LABELS.get(run, run)}\n{c}: p={probs[classes.index(c)]:.2f}", fontsize=8)
        for ax in row:
            ax.axis("off")
    fig.suptitle("Grad-CAM: where does each EfficientNet look for the true disease?", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "gradcam_b0_vs_b3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="EfficientNet B0 vs B3 report figures")
    parser.add_argument("--runs", nargs="+", default=["efficientnet", "efficientnet_b3"], choices=list(CONFIGS))
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--no-gradcam", action="store_true")
    args = parser.parse_args()

    out = RESULTS / "efficientnet_report"
    out.mkdir(parents=True, exist_ok=True)
    scoreboard(out)
    curves(args.runs, out)
    per_class(args.runs, out)
    if not args.no_gradcam:
        gradcam_compare(args.runs, args.num_images, out)
    print(f"[report] figures saved to {out}")


if __name__ == "__main__":
    main()
