"""Step 4: Exploratory Data Analysis (EDA) for NIH ChestX-ray14.

Run after downloading the data (and ideally after creating the splits):
    python -m src.eda

Creates report-ready figures + tables in results/eda/ :
    01_label_distribution_full.png   all 15 NIH labels, our selected classes highlighted
    02_class_distribution_subset.png images per class in the scoped, balanced subset
    03_label_cooccurrence.png        how often two diseases appear on the same X-ray
    04_labels_per_image.png          multi-label evidence (1, 2, 3+ labels per image)
    05_sample_images.png             example X-rays for every class
    06_images_per_patient.png        why we MUST split by patient (leakage risk)
    07_demographics.png              age and gender
    08_view_position.png             PA vs AP per class (possible shortcut for the models)
    09_split_distribution.png        class prevalence in train / val / test
    eda_summary.json                 key numbers + data-quality issues for Report Section 3
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.config import BASE_CONFIG, load_config
from src.preprocessing.splits import _RENAME, load_labels

# Colours (validated colour-blind-safe reference palette)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MUTED = "#b8b7b0"
INK, INK2 = "#0b0b0b", "#52514e"
SPLIT_COLORS = {"train": BLUE, "val": ORANGE, "test": AQUA}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def _save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] saved {name}")


def _hbar(ax, labels, values, colors, total=None):
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.7, edgecolor="white", linewidth=2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        txt = f"{int(v):,}" + (f"  ({100 * v / total:.1f}%)" if total else "")
        ax.text(v, yi, "  " + txt, va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, max(values) * 1.25)


# --------------------------------------------------------------------------------------
def label_distribution_full(raw: pd.DataFrame, classes, out: Path) -> dict:
    counts = raw["labels"].str.split("|").explode().value_counts()
    colors = [BLUE if c in classes else MUTED for c in counts.index]
    fig, ax = plt.subplots(figsize=(9, 6))
    _hbar(ax, counts.index, counts.values, colors, total=len(raw))
    ax.set_title("NIH ChestX-ray14 – images per label (full dataset)")
    ax.set_xlabel("number of images (an image can have several labels)")
    ax.text(0.99, 0.02, "blue = selected for ChestNet   grey = not used", transform=ax.transAxes,
            ha="right", fontsize=8, color=INK2)
    _save(fig, out, "01_label_distribution_full.png")
    return counts.to_dict()


def class_distribution_subset(df: pd.DataFrame, classes, out: Path) -> dict:
    counts = df[classes].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _hbar(ax, counts.index, counts.values, BLUE, total=len(df))
    ax.set_title(f"Scoped subset – images per class (n = {len(df):,} images)")
    ax.set_xlabel("number of images")
    _save(fig, out, "02_class_distribution_subset.png")
    return {k: int(v) for k, v in counts.items()}


def cooccurrence(df: pd.DataFrame, classes, out: Path):
    y = df[classes].to_numpy()
    co = y.T @ y                                    # co[i, j] = images with class i AND j
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(co, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(co):
        ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=8,
                color="white" if v > co.max() * 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="images")
    ax.set_title("Label co-occurrence (diagonal = total per class)")
    _save(fig, out, "03_label_cooccurrence.png")
    return pd.DataFrame(co, index=classes, columns=classes)


def labels_per_image(df: pd.DataFrame, classes, out: Path) -> dict:
    n = df[classes].sum(axis=1).clip(upper=4)
    counts = n.value_counts().sort_index()
    names = [f"{k}" if k < 4 else "4+" for k in counts.index]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(names, counts.values, color=BLUE, width=0.6, edgecolor="white", linewidth=2)
    ax.grid(axis="x", visible=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{100 * v / len(df):.1f}%", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_title("Labels per image → a multi-label problem")
    ax.set_xlabel("number of selected labels on one X-ray")
    ax.set_ylabel("images")
    _save(fig, out, "04_labels_per_image.png")
    return {str(k): int(v) for k, v in zip(names, counts.values)}


def sample_images(df: pd.DataFrame, classes, out: Path, per_class: int = 3, seed: int = 42):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(classes), per_class, figsize=(2.2 * per_class, 2.3 * len(classes)))
    for r, c in enumerate(classes):
        pool = df[df[c] == 1]
        # Prefer single-label examples so the picture really shows that class.
        single = pool[pool[classes].sum(axis=1) == 1]
        pool = single if len(single) >= per_class else pool
        picks = pool.iloc[rng.choice(len(pool), size=min(per_class, len(pool)), replace=False)]
        for k in range(per_class):
            ax = axes[r, k]
            ax.axis("off")
            if k < len(picks):
                img = Image.open(picks.iloc[k]["path"]).convert("L")
                ax.imshow(img, cmap="gray")
        axes[r, 0].set_title(c, loc="left", fontsize=10)
    fig.suptitle("Sample chest X-rays per class", fontweight="bold")
    _save(fig, out, "05_sample_images.png")


def images_per_patient(raw: pd.DataFrame, out: Path) -> dict:
    per_patient = raw.groupby("patient_id").size()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(per_patient.clip(upper=30), bins=30, color=BLUE, edgecolor="white", linewidth=1)
    ax.set_yscale("log")
    ax.set_title("Images per patient (full dataset) – why we split by patient")
    ax.set_xlabel("images per patient (30 = 30 or more)")
    ax.set_ylabel("patients (log scale)")
    ax.grid(axis="x", visible=False)
    _save(fig, out, "06_images_per_patient.png")
    return {"patients": int(per_patient.size), "mean": round(float(per_patient.mean()), 2),
            "median": float(per_patient.median()), "max": int(per_patient.max()),
            "patients_with_multiple_images_pct": round(100 * float((per_patient > 1).mean()), 1)}


def demographics(df: pd.DataFrame, out: Path) -> dict:
    if "age" not in df or "gender" not in df:
        return {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 1]})
    axes[0].hist(df["age"].dropna(), bins=range(0, 101, 5), color=BLUE, edgecolor="white", linewidth=1)
    axes[0].set_title("Patient age (subset)")
    axes[0].set_xlabel("age (years)")
    axes[0].set_ylabel("images")
    axes[0].grid(axis="x", visible=False)
    g = df["gender"].value_counts()
    axes[1].bar(g.index, g.values, color=BLUE, width=0.6, edgecolor="white", linewidth=2)
    axes[1].set_title("Gender")
    axes[1].grid(axis="x", visible=False)
    for i, v in enumerate(g.values):
        axes[1].text(i, v, f"{100 * v / g.sum():.0f}%", ha="center", va="bottom", fontsize=9, color=INK2)
    _save(fig, out, "07_demographics.png")
    return {"age_mean": round(float(df["age"].mean()), 1), "age_median": float(df["age"].median()),
            "gender": {k: int(v) for k, v in g.items()}}


def view_position(df: pd.DataFrame, classes, out: Path) -> dict:
    """AP images are usually taken of sicker, bed-bound patients. If a class is mostly AP,
    a model could learn the VIEW instead of the disease (shortcut learning)."""
    if "view" not in df:
        return {}
    ap = {c: 100 * float((df.loc[df[c] == 1, "view"] == "AP").mean()) for c in classes}
    ap = dict(sorted(ap.items(), key=lambda kv: -kv[1]))
    overall = 100 * float((df["view"] == "AP").mean())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _hbar(ax, list(ap), list(ap.values()), BLUE)
    for t in ax.texts:
        t.set_text(t.get_text().strip() + "%")
    ax.axvline(overall, color=INK2, linestyle="--", linewidth=1.2)
    ax.text(overall, len(ap) - 0.4, f" average {overall:.0f}%", fontsize=8, color=INK2, va="bottom")
    ax.set_title("Share of AP (bedside) views per class – possible shortcut signal")
    ax.set_xlabel("% of images taken in AP view (rest are PA)")
    _save(fig, out, "08_view_position.png")
    return {k: round(v, 1) for k, v in ap.items()} | {"overall_AP_pct": round(overall, 1)}


def split_distribution(splits_dir: Path, classes, out: Path) -> dict:
    files = {n: splits_dir / f"{n}.csv" for n in ("train", "val", "test")}
    if not all(f.exists() for f in files.values()):
        print("[eda] splits not found – run `python -m src.preprocessing.splits` first to get figure 09")
        return {}
    parts = {n: pd.read_csv(f) for n, f in files.items()}
    prev = pd.DataFrame({n: 100 * p[classes].mean() for n, p in parts.items()})
    x = np.arange(len(classes))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for k, (n, col) in enumerate(prev.items()):
        ax.bar(x + (k - 1) * w, col.values, width=w, color=SPLIT_COLORS[n], edgecolor="white", linewidth=1.5,
               label=f"{n} ({len(parts[n]):,} images, {parts[n].patient_id.nunique():,} patients)")
    ax.set_xticks(x, classes, rotation=30, ha="right")
    ax.set_ylabel("% of images with the label")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Class prevalence per split (patient-level, label-balanced)")
    _save(fig, out, "09_split_distribution.png")
    return prev.round(2).to_dict()


# --------------------------------------------------------------------------------------
def run_eda(cfg: dict, n_samples: int = 3) -> dict:
    classes = cfg["data"]["classes"]
    out = Path(cfg["output"]["results_dir"]) / "eda"
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(cfg["data"]["csv_path"]).rename(columns=_RENAME)
    splits_dir = Path(cfg["data"]["splits_dir"])
    if (splits_dir / "train.csv").exists():                      # the scoped subset actually used
        df = pd.concat([pd.read_csv(splits_dir / f"{n}.csv") for n in ("train", "val", "test")],
                       ignore_index=True)
        scope = "scoped subset (from data/splits)"
    else:                                                          # before splitting: all matching images
        df = load_labels(cfg)
        scope = "all images with a selected label (no splits yet)"

    summary = {
        "dataset": "NIH ChestX-ray14",
        "full_dataset": {"images": int(len(raw)), "patients": int(raw["patient_id"].nunique())},
        "analysed": {"scope": scope, "images": int(len(df)), "patients": int(df["patient_id"].nunique())},
        "label_counts_full": label_distribution_full(raw, classes, out),
        "class_counts_analysed": class_distribution_subset(df, classes, out),
        "labels_per_image": labels_per_image(df, classes, out),
        "images_per_patient_full": images_per_patient(raw, out),
        "demographics": demographics(df, out),
        "ap_view_pct_per_class": view_position(df, classes, out),
        "split_prevalence_pct": split_distribution(splits_dir, classes, out),
    }
    cooccurrence(df, classes, out).to_csv(out / "label_cooccurrence.csv")
    if "path" in df.columns:
        sample_images(df, classes, out, per_class=n_samples, seed=cfg["seed"])

    # Image format (from the label file; NIH images are 8-bit grayscale PNG).
    if {"orig_width", "orig_height"} <= set(raw.columns):
        summary["original_image_size"] = {
            "width_min_max": [int(raw.orig_width.min()), int(raw.orig_width.max())],
            "height_min_max": [int(raw.orig_height.min()), int(raw.orig_height.max())],
            "most_common": raw.groupby(["orig_width", "orig_height"]).size().idxmax(),
        }
        summary["original_image_size"]["most_common"] = "x".join(map(str, summary["original_image_size"]["most_common"]))

    # Data-quality issues to discuss in Report Section 3.
    ages = pd.read_csv(cfg["data"]["csv_path"])["Patient Age"] if "Patient Age" in pd.read_csv(cfg["data"]["csv_path"], nrows=1) else None
    counts = summary["class_counts_analysed"]
    summary["data_quality_issues"] = {
        "label_noise": "Labels were text-mined from radiology reports with NLP (the NIH paper reports "
                       "roughly 90% accuracy), so some labels are wrong.",
        "class_imbalance_ratio_max_min": round(max(counts.values()) / max(1, min(counts.values())), 2),
        "invalid_ages_over_100": int((ages > 100).sum()) if ages is not None else None,
        "patients_with_multiple_images_pct": summary["images_per_patient_full"]["patients_with_multiple_images_pct"],
        "view_position_confounder": "AP (bedside) views are more frequent for some diseases – a possible shortcut.",
    }
    with open(out / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[eda] done – figures and eda_summary.json in {out}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exploratory data analysis for ChestNet")
    parser.add_argument("--config", default=str(BASE_CONFIG))
    parser.add_argument("--samples", type=int, default=3, help="sample images per class")
    args = parser.parse_args()
    run_eda(load_config(args.config), n_samples=args.samples)
