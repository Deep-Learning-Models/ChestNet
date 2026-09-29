"""Create Grad-CAM heatmaps for a CNN model on test images.

Usage:
    python -m src.explain --config configs/resnet50.yaml --num-images 8
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.models import build_model, gradcam_layer
from src.preprocessing.dataset import _load_image
from src.preprocessing.splits import load_splits
from src.utils.gradcam import gradcam_heatmap
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM heatmaps for ChestNet CNNs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--num-images", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    name = cfg["model"]["name"]
    layer = gradcam_layer(cfg)
    if layer is None:
        raise SystemExit(f"Grad-CAM is only set up for the CNN models, not '{name}'.")
    classes = cfg["data"]["classes"]
    ckpt = args.checkpoint or Path(cfg["output"]["checkpoint_dir"]) / f"{name}_best.weights.h5"

    model = build_model(cfg)
    model.load_weights(ckpt)

    _, _, test_df = load_splits(cfg)
    diseased = test_df[test_df["No Finding"] == 0] if "No Finding" in test_df else test_df
    sample = diseased.sample(min(args.num_images, len(diseased)), random_state=cfg["seed"])

    fig, axes = plt.subplots(len(sample), 2, figsize=(7, 3.4 * len(sample)))
    axes = np.atleast_2d(axes)
    for row, (_, r) in zip(axes, sample.iterrows()):
        img = _load_image(r["path"], cfg["data"]["image_size"]).numpy()
        probs = model.predict(img[None], verbose=0)[0]
        c = int(np.argmax(probs))
        heat = gradcam_heatmap(model, img, layer, c)
        true = ", ".join(k for k in classes if r[k] == 1)

        row[0].imshow(img.astype("uint8"))
        row[0].set_title(f"True: {true}", fontsize=9)
        row[1].imshow(img.astype("uint8"))
        row[1].imshow(heat, cmap="jet", alpha=0.4)
        row[1].set_title(f"Pred: {classes[c]} ({probs[c]:.2f})", fontsize=9)
        for ax in row:
            ax.axis("off")

    out = Path(cfg["output"]["results_dir"]) / name / "gradcam.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[explain] saved {out}")


if __name__ == "__main__":
    main()
