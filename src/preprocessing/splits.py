"""Steps 3 + 5: scope the NIH ChestX-ray14 subset and split it BY PATIENT.

Run once (all four models then share the same CSVs):
    python -m src.preprocessing.splits

What it does
------------
1. Reads the label CSV (configs/base.yaml -> data.csv_path) and turns "Finding Labels" into one 0/1 column per
   selected class (multi-label).
2. Cleans the metadata (impossible ages, label conflicts) and keeps it for the EDA.
3. Builds a balanced subset (up to N images per class, rarest class first).
4. Splits by PATIENT ID (70/15/15) so no patient is in two splits. Many random
   patient splits are tried and the one whose label distribution best matches the
   whole subset is kept ("label-balanced group split"). This keeps rare diseases
   represented in validation and test.
5. Saves train/val/test CSVs, a class-count summary and a leakage report (JSON).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import BASE_CONFIG, load_config

# Column names in the official NIH / Kaggle label file -> short names used in the code.
_RENAME = {
    "Image Index": "image",
    "Finding Labels": "labels",
    "Follow-up #": "follow_up",
    "Patient ID": "patient_id",
    "Patient Age": "age",
    "Patient Gender": "gender",
    "View Position": "view",
    "OriginalImage[Width": "orig_width",       # full dataset (Data_Entry_2017.csv)
    "Height]": "orig_height",
    "OriginalImageWidth": "orig_width",         # Kaggle random sample (sample_labels.csv)
    "OriginalImageHeight": "orig_height",
}
META_COLS = ["follow_up", "age", "gender", "view", "orig_width", "orig_height"]


# --------------------------------------------------------------------------------------
# Step 3 – load, clean and scope
# --------------------------------------------------------------------------------------
def parse_age(age: pd.Series) -> pd.Series:
    """Patient age in years.

    The full label file stores numbers (58); the Kaggle random sample stores text such
    as '058Y' (a few may use 'M' = months or 'D' = days). Both become numbers here.
    """
    if pd.api.types.is_numeric_dtype(age):
        return age.astype(float)
    s = age.astype(str).str.strip().str.upper()
    number = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
    unit = s.str.extract(r"([YMD])$")[0].fillna("Y")
    return number / unit.map({"Y": 1.0, "M": 12.0, "D": 365.0})


def _index_images(image_dir: Path) -> dict:
    """Map image file name -> full path (Kaggle stores images in images_001 ... images_012)."""
    # as_posix() writes "data/raw/..." with forward slashes, so split CSVs made on Windows
    # also work on Linux / Google Colab (and the other way round).
    return {p.name: p.as_posix() for p in image_dir.rglob("*.png")}


def load_labels(cfg: dict, check_files: bool = True) -> pd.DataFrame:
    """Read the full label file, add one 0/1 column per selected class and clean metadata.

    Returns ALL images that carry at least one selected class (before balancing).
    Used by both the split script and the EDA script.
    """
    data_cfg = cfg["data"]
    classes = data_cfg["classes"]

    df = pd.read_csv(data_cfg["csv_path"])
    df = df.rename(columns=_RENAME)
    keep = ["image", "labels", "patient_id"] + [c for c in META_COLS if c in df.columns]
    df = df[keep].copy()

    # Multi-label encoding: one 0/1 column per selected class.
    label_sets = df["labels"].str.split("|")
    for c in classes:
        df[c] = label_sets.apply(lambda ls, c=c: int(c in ls))

    # Data cleaning --------------------------------------------------------------
    # (a) "No Finding" must never co-occur with a disease label.
    if "No Finding" in classes:
        diseases = [c for c in classes if c != "No Finding"]
        conflict = (df["No Finding"] == 1) & (df[diseases].sum(axis=1) > 0)
        df.loc[conflict, "No Finding"] = 0
    # (b) Ages as numbers; a few NIH ages are impossible (e.g. 414 years) -> treat as missing.
    if "age" in df.columns:
        df["age"] = parse_age(df["age"])
        df["age_invalid"] = df["age"] > 100
        df.loc[df["age_invalid"], "age"] = np.nan

    # Keep images that have at least one of the selected classes.
    # (An image labelled only with a non-selected disease, e.g. "Hernia", is dropped.)
    df = df[df[classes].sum(axis=1) > 0].reset_index(drop=True)

    if check_files:
        index = _index_images(Path(data_cfg["image_dir"]))
        df["path"] = df["image"].map(index)
        missing = int(df["path"].isna().sum())
        df = df.dropna(subset=["path"]).reset_index(drop=True)
        if missing:
            print(f"[splits] {missing} labelled images not found in {data_cfg['image_dir']} – skipped")
    return df


def build_subset(cfg: dict, require_files: bool = True) -> pd.DataFrame:
    """Balanced subset: up to N images per class, filling the rarest class first.

    Because an image can carry several labels, a common class (e.g. Infiltration) is
    often already partly filled by images picked for rarer classes – those count too.

    The images are chosen from the LABEL FILE ONLY (labels + seed), never from which
    PNG files happen to be on disk. So every machine (laptop, Google Colab, a team
    member's PC) picks exactly the same images, and on Colab only these images need to
    be extracted from the 42 GB download (see `--list-only`).
    require_files=False -> return the selection without checking the image files.
    """
    classes = cfg["data"]["classes"]
    n = cfg["data"]["max_images_per_class"]
    rng = np.random.default_rng(cfg["seed"])
    df = load_labels(cfg, check_files=False)

    chosen = np.zeros(len(df), dtype=bool)
    for c in sorted(classes, key=lambda c: df[c].sum()):          # rarest first
        already = int(df.loc[chosen, c].sum())
        pool = np.flatnonzero((df[c].to_numpy() == 1) & ~chosen)
        take = max(0, min(n - already, len(pool)))
        if take:
            chosen[rng.choice(pool, size=take, replace=False)] = True
    subset = df[chosen].reset_index(drop=True)
    if not require_files:
        return subset

    image_dir = Path(cfg["data"]["image_dir"])
    subset["path"] = subset["image"].map(_index_images(image_dir))
    missing = subset.loc[subset["path"].isna(), "image"]
    if len(missing):
        raise FileNotFoundError(
            f"[splits] {len(missing)} of the {len(subset)} selected images are not in {image_dir} "
            f"(e.g. {missing.iloc[0]}). Every model must use the same images, so download them first. "
            f"Only the selected images are needed: 'python -m src.preprocessing.splits --list-only' "
            f"writes their names to {cfg['data']['splits_dir']}/subset_images.txt.")
    return subset


def write_subset_list(cfg: dict) -> Path:
    """Save the names of the selected images (no image files needed).

    Used on Google Colab: only these images are extracted from the full dataset ZIP.
    """
    classes = cfg["data"]["classes"]
    subset = build_subset(cfg, require_files=False)
    out = Path(cfg["data"]["splits_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "subset_images.txt"
    path.write_text("\n".join(subset["image"]) + "\n", encoding="utf-8")
    print(f"[splits] {len(subset)} images selected ({subset.patient_id.nunique()} patients) -> {path}")
    print(subset[classes].sum().to_string())
    return path


# --------------------------------------------------------------------------------------
# Step 5 – patient-level, label-balanced split
# --------------------------------------------------------------------------------------
def _group_split(df: pd.DataFrame, val_size: float, test_size: float, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trval_idx, test_idx = next(gss.split(df, groups=df["patient_id"]))
    trval = df.iloc[trval_idx]
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size / (1.0 - test_size), random_state=seed)
    tr_idx, va_idx = next(gss.split(trval, groups=trval["patient_id"]))
    return trval.iloc[tr_idx], trval.iloc[va_idx], df.iloc[test_idx]


def _split_score(parts, df, classes, sizes, min_pos) -> float:
    """Lower is better: label-distribution mismatch + image-ratio mismatch + rare-class penalty."""
    overall = df[classes].mean().to_numpy()
    score = 0.0
    for part, target in zip(parts, sizes):
        score += np.abs(part[classes].mean().to_numpy() - overall).sum()
        score += abs(len(part) / len(df) - target)
        score += 10.0 * int((part[classes].sum().to_numpy() < min_pos).any())
    return score


def patient_split(df: pd.DataFrame, cfg: dict):
    """Try many patient-level splits and keep the most label-balanced one.

    Every candidate is a pure GroupShuffleSplit on Patient ID, so leakage is impossible
    by construction; the search only chooses WHICH patients go where.
    """
    data_cfg = cfg["data"]
    classes = data_cfg["classes"]
    val_size, test_size = data_cfg["val_size"], data_cfg["test_size"]
    sizes = (1.0 - val_size - test_size, val_size, test_size)
    n_candidates = int(data_cfg.get("split_candidates", 200))
    min_pos = int(data_cfg.get("min_positives_per_split", 30))

    rng = np.random.default_rng(cfg["seed"])
    best, best_score, best_seed = None, np.inf, None
    for cand_seed in rng.integers(0, 2**31 - 1, size=n_candidates):
        parts = _group_split(df, val_size, test_size, int(cand_seed))
        s = _split_score(parts, df, classes, sizes, min_pos)
        if s < best_score:
            best, best_score, best_seed = parts, s, int(cand_seed)

    train, val, test = (p.reset_index(drop=True) for p in best)
    report = leakage_report(train, val, test)
    report.update({"chosen_split_seed": best_seed, "balance_score": round(float(best_score), 4),
                   "candidates_tried": n_candidates})
    assert report["patient_overlap_total"] == 0, "Patient leakage between splits!"
    assert report["image_overlap_total"] == 0, "Image appears in more than one split!"
    return train, val, test, report


def leakage_report(train, val, test) -> dict:
    """Evidence for the report that no patient or image is shared between splits."""
    p = {n: set(d["patient_id"]) for n, d in [("train", train), ("val", val), ("test", test)]}
    i = {n: set(d["image"]) for n, d in [("train", train), ("val", val), ("test", test)]}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    rep = {f"patient_overlap_{a}_{b}": len(p[a] & p[b]) for a, b in pairs}
    rep.update({f"image_overlap_{a}_{b}": len(i[a] & i[b]) for a, b in pairs})
    rep["patient_overlap_total"] = sum(rep[f"patient_overlap_{a}_{b}"] for a, b in pairs)
    rep["image_overlap_total"] = sum(rep[f"image_overlap_{a}_{b}"] for a, b in pairs)
    return rep


def make_splits(cfg: dict) -> None:
    out = Path(cfg["data"]["splits_dir"])
    out.mkdir(parents=True, exist_ok=True)
    classes = cfg["data"]["classes"]

    df = build_subset(cfg)
    train, val, test, report = patient_split(df, cfg)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(out / f"{name}.csv", index=False)

    parts = [("train", train), ("val", val), ("test", test)]
    summary = pd.DataFrame({n: p[classes].sum() for n, p in parts})
    summary.loc["images"] = [len(p) for _, p in parts]
    summary.loc["patients"] = [p.patient_id.nunique() for _, p in parts]
    summary["total"] = summary.sum(axis=1)
    summary.to_csv(out / "summary.csv")

    report.update({f"{n}_images": len(p) for n, p in parts})
    report.update({f"{n}_patients": int(p.patient_id.nunique()) for n, p in parts})
    with open(out / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[splits] Saved to", out)
    print(summary)
    print(f"[splits] Leakage check: {report['patient_overlap_total']} shared patients, "
          f"{report['image_overlap_total']} shared images ✅")


def load_splits(cfg: dict):
    out = Path(cfg["data"]["splits_dir"])
    if not (out / "train.csv").exists():
        make_splits(cfg)
    return tuple(pd.read_csv(out / f"{n}.csv") for n in ("train", "val", "test"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create patient-level train/val/test splits")
    parser.add_argument("--config", default=str(BASE_CONFIG))
    parser.add_argument("--list-only", action="store_true",
                        help="only save the names of the selected images (no image files needed)")
    args = parser.parse_args()
    if args.list_only:
        write_subset_list(load_config(args.config))
    else:
        make_splits(load_config(args.config))
