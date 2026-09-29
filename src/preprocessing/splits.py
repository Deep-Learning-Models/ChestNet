"""Build the scoped, balanced NIH ChestX-ray14 subset and split it BY PATIENT.

Run once:  python -m src.preprocessing.splits --config configs/base.yaml
The resulting CSVs in data/splits/ are shared by all four models.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import BASE_CONFIG, load_config


def _index_images(image_dir: Path) -> dict:
    """Map image file name -> full path (Kaggle stores images in images_001 ... images_012)."""
    return {p.name: str(p) for p in image_dir.rglob("*.png")}


def build_subset(cfg: dict) -> pd.DataFrame:
    data_cfg = cfg["data"]
    classes = data_cfg["classes"]
    rng = np.random.default_rng(cfg["seed"])

    df = pd.read_csv(data_cfg["csv_path"])
    df = df.rename(columns={"Image Index": "image", "Finding Labels": "labels", "Patient ID": "patient_id"})
    df = df[["image", "labels", "patient_id"]]

    # One 0/1 column per selected class (multi-label).
    label_sets = df["labels"].str.split("|")
    for c in classes:
        df[c] = label_sets.apply(lambda ls, c=c: int(c in ls))
    df = df[df[classes].sum(axis=1) > 0].reset_index(drop=True)

    # Keep only images that exist locally.
    index = _index_images(Path(data_cfg["image_dir"]))
    df["path"] = df["image"].map(index)
    missing = df["path"].isna().sum()
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    if missing:
        print(f"[splits] {missing} labelled images not found in {data_cfg['image_dir']} – skipped")

    # Balanced subset: sample up to N images for each class (rarest class first).
    n = data_cfg["max_images_per_class"]
    chosen = set()
    for c in sorted(classes, key=lambda c: df[c].sum()):
        pool = df.index[(df[c] == 1) & (~df.index.isin(chosen))].to_numpy()
        already = int(df.loc[list(chosen), c].sum()) if chosen else 0
        take = max(0, min(n - already, len(pool)))
        if take:
            chosen.update(rng.choice(pool, size=take, replace=False).tolist())
    return df.loc[sorted(chosen)].reset_index(drop=True)


def patient_split(df: pd.DataFrame, cfg: dict):
    """Split by patient so no patient appears in more than one split (prevents leakage)."""
    seed = cfg["seed"]
    val_size, test_size = cfg["data"]["val_size"], cfg["data"]["test_size"]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss.split(df, groups=df["patient_id"]))
    trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]

    rel_val = val_size / (1.0 - test_size)
    gss = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    train_idx, val_idx = next(gss.split(trainval, groups=trainval["patient_id"]))
    train, val = trainval.iloc[train_idx], trainval.iloc[val_idx]

    # Safety check: no patient overlap.
    p_tr, p_va, p_te = set(train.patient_id), set(val.patient_id), set(test.patient_id)
    assert not (p_tr & p_va or p_tr & p_te or p_va & p_te), "Patient leakage between splits!"
    return train, val, test


def make_splits(cfg: dict) -> None:
    out = Path(cfg["data"]["splits_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df = build_subset(cfg)
    train, val, test = patient_split(df, cfg)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(out / f"{name}.csv", index=False)

    classes = cfg["data"]["classes"]
    summary = pd.DataFrame({n: p[classes].sum() for n, p in [("train", train), ("val", val), ("test", test)]})
    summary.loc["images"] = [len(train), len(val), len(test)]
    summary.loc["patients"] = [train.patient_id.nunique(), val.patient_id.nunique(), test.patient_id.nunique()]
    summary.to_csv(out / "summary.csv")
    print("[splits] Saved to", out)
    print(summary)


def load_splits(cfg: dict):
    out = Path(cfg["data"]["splits_dir"])
    if not (out / "train.csv").exists():
        make_splits(cfg)
    return tuple(pd.read_csv(out / f"{n}.csv") for n in ("train", "val", "test"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create patient-level train/val/test splits")
    parser.add_argument("--config", default=str(BASE_CONFIG))
    args = parser.parse_args()
    make_splits(load_config(args.config))
