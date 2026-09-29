# 📊 Dataset: NIH ChestX-ray14

The images are **not** stored in this repository (about 45 GB). Follow these steps to get them.

## 1. Source

| Item | Detail |
|------|--------|
| Name | NIH ChestX-ray14 (ChestX-ray8 extended) |
| Creator | National Institutes of Health (NIH) Clinical Center |
| Download | https://www.kaggle.com/datasets/nih-chest-xrays/data |
| Original release | https://nihcc.app.box.com/v/ChestXray-NIHCC |
| Paper | Wang et al., *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks*, CVPR 2017 |
| Size | 112,120 frontal-view X-rays (1024 × 1024 PNG) from 30,805 patients |
| Labels | 14 diseases + *No Finding*, text-mined from radiology reports (multi-label) |

## 2. Download

**Option A: Kaggle website.** Download the ZIP from the link above and extract it.

**Option B: Kaggle CLI**

```bash
pip install kaggle
kaggle datasets download -d nih-chest-xrays/data -p data/ --unzip
```

## 3. Folder layout

After extracting, the files should look like this:

```text
data/
├── Data_Entry_2017.csv        # labels (one row per image)
├── raw/
│   ├── images_001/images/*.png
│   ├── images_002/images/*.png
│   └── ...                    # up to images_012
└── splits/                    # created by the split script
```

Move the `images_0XX` folders into `data/raw/`. Images are found recursively, so the exact sub-folder names do not matter.

## 4. Create the splits

```bash
python -m src.preprocessing.splits
```

This script:

- keeps the 7 selected diseases and *No Finding* (set in `configs/base.yaml`),
- samples up to 1,500 images per class for a balanced subset,
- splits **by patient ID** (70 / 15 / 15) so no patient appears in two splits,
- saves `train.csv`, `val.csv`, `test.csv` and `summary.csv` in `data/splits/`.

## 5. Known data-quality issues

- **Label noise:** labels were extracted from reports by NLP, so some are wrong.
- **Class imbalance:** diseases such as Pneumonia are rare, which is why training uses a weighted loss.
- **Multiple images per patient:** handled by the patient-level split.
