# 📊 Dataset: Random Sample of NIH Chest X-ray Dataset

The images are **not** stored in this repository. Follow these steps to get them.

## 1. Source

| Item | Detail |
|------|--------|
| Name | Random Sample of NIH Chest X-ray Dataset |
| Download | https://www.kaggle.com/datasets/nih-chest-xrays/sample |
| Parent dataset | NIH ChestX-ray14 (112,120 images): https://www.kaggle.com/datasets/nih-chest-xrays/data |
| Creator | National Institutes of Health (NIH) Clinical Center |
| Paper | Wang et al., *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks*, CVPR 2017 |
| Size | 5,606 frontal-view X-rays (1024 × 1024 PNG), randomly sampled from the full dataset |
| Labels | 14 diseases + *No Finding*, text-mined from radiology reports (multi-label), in `sample_labels.csv` |

## 2. Download

**Option A: Kaggle website.** Download the ZIP from the link above and extract it.

**Option B: Kaggle CLI**

```bash
pip install kaggle
kaggle datasets download -d nih-chest-xrays/sample -p data/raw --unzip
```

## 3. Folder layout

After extracting, the files should look like this:

```text
data/
├── sample_labels.csv          # labels (one row per image) – move it here
├── raw/
│   └── .../images/*.png       # the 5,606 X-rays (any sub-folder)
└── splits/                    # created by the split script
```

Images are found recursively, so the exact sub-folder names do not matter.

## 4. Create the splits

```bash
python -m src.preprocessing.splits
```

This script:

- keeps the 7 selected diseases and *No Finding* (set in `configs/base.yaml`),
- uses every image of the 7 diseases and caps *No Finding* at 1,500 images,
- splits **by patient ID** (70 / 15 / 15) so no patient appears in two splits, keeping the most label-balanced of 200 candidate splits,
- saves `train.csv`, `val.csv`, `test.csv`, `summary.csv` and `split_report.json` (leakage check) in `data/splits/`.

Then run the EDA (`python -m src.eda`) and the pipeline check (`python -m src.check_pipeline`).

> 💡 To use the full dataset instead, set `data.csv_path` in `configs/base.yaml` to `data/Data_Entry_2017.csv`.

## 5. Known data-quality issues

- **Small rare classes:** the sample has only a few dozen Pneumonia images and about 140 Cardiomegaly images, so their test-set metrics are uncertain. Report this as a limitation.
- **Label noise:** labels were extracted from reports by NLP, so some are wrong.
- **Class imbalance:** most images are *No Finding* or Infiltration, which is why training uses a weighted loss.
- **Multiple images per patient:** handled by the patient-level split.
- **Age format:** the sample stores ages as text (e.g. `058Y`); the code converts them to years.
- **Impossible ages:** a few records may have ages above 100; they are treated as missing.
- **"No Finding" conflicts:** "No Finding" is removed if an image also has a disease label.
- **View position (AP vs PA):** bedside AP images are more common for sicker patients, which a model could use as a shortcut (see `results/eda/08_view_position.png`).
