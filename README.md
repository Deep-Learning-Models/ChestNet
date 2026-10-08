<p align="center">
  <img src="docs/assets/banner.png" alt="ChestNet banner" width="100%">
</p>

# 🫁 ChestNet

**Multi-Model Deep Learning Framework for Chest X-ray Multi-Disease Classification**

![Status](https://img.shields.io/badge/status-pipeline%20ready-2ea44f)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3-D00000?logo=keras&logoColor=white)
![Dataset](https://img.shields.io/badge/dataset-NIH%20Chest%20X--ray%20(random%20sample)-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**Current status:** 🚧 PIPELINE READY. Code for all four models, the shared configs and the random seeds is in place. Training runs and results are in progress.

ChestNet trains and critically compares **four architecturally distinct deep learning models** on chest X-ray images under the same data splits, augmentation and evaluation protocol. Each model predicts the presence of one or more thoracic diseases (for example Effusion, Infiltration, Atelectasis, Cardiomegaly, Nodule, Mass, Pneumonia or No Finding). Grad-CAM heatmaps show *where* the CNNs look when they make a prediction.

> ⚠️ **Disclaimer:** this is an academic research project (SE4050 – Deep Learning). It is a decision-support concept and **not** a medical diagnostic tool.

---

## 🧭 System overview

<p align="center">
  <img src="docs/assets/system-diagram.jpeg" alt="ChestNet system architecture" width="90%">
</p>

---

## 🧠 The four models

| # | Model | Role | Key idea |
|---|-------|------|----------|
| 1 | 🟧 **Custom CNN** (from scratch) | Baseline | Conv → BatchNorm → ReLU → MaxPool blocks, dropout and a dense multi-label head. No pretrained weights, so it shows the value of transfer learning. |
| 2 | 🟥 **ResNet50** (transfer learning) | Deep residual benchmark | ImageNet-pretrained and fine-tuned. Skip connections keep gradients flowing through deep networks. |
| 3 | 🟩 **EfficientNet-B0/B3** | Efficiency-focused CNN | Compound scaling of depth, width and resolution. Compared on accuracy per parameter and per second. |
| 4 | 🟪 **Vision Transformer (ViT-B/16)** | Attention-based | Image patches become tokens, and self-attention captures global context across both lungs. Uses ImageNet weights from `keras-hub`; a smaller from-scratch ViT is available with `pretrained: false`. |

---

## 🛠️ Technology

- **Language:** Python 3.10+
- **Deep learning:** TensorFlow 2.16+ with Keras 3 (`keras.applications` for ResNet50 and EfficientNet, `keras-hub` for the pretrained ViT-B/16)
- **Input pipeline:** `tf.data` with Keras augmentation layers
- **Data and metrics:** NumPy, pandas, scikit-learn
- **Visualisation:** Matplotlib, Seaborn, Grad-CAM (implemented with `tf.GradientTape`)
- **Configuration:** YAML files in `configs/` with a fixed random seed
- **Workflow:** Jupyter notebooks for EDA, `src/` for reusable, runnable code

---

## 📁 Project structure

```text
ChestNet/
├── configs/
│   ├── base.yaml            # Shared settings for ALL models (seed, classes, splits, training)
│   ├── custom_cnn.yaml      # Model-specific overrides
│   ├── resnet50.yaml
│   ├── efficientnet.yaml
│   └── vit.yaml
├── data/
│   ├── README.md            # How to download and place the dataset
│   ├── raw/                 # NIH images (git-ignored)
│   └── splits/              # Saved patient-level train/val/test CSVs
├── docs/assets/             # Banner and system diagram
├── notebooks/               # EDA and exploration
├── results/                 # Metrics, curves, ROC plots, confusion matrices, Grad-CAM
├── saved_models/            # Best weights per model (git-ignored)
├── src/
│   ├── config.py            # Merges base.yaml with a model config
│   ├── train.py             # Train one model
│   ├── evaluate.py          # Test-set metrics and comparison table
│   ├── explain.py           # Grad-CAM heatmaps
│   ├── models/              # custom_cnn.py, resnet50.py, efficientnet.py, vit.py
│   ├── preprocessing/       # splits.py (patient-level split), dataset.py (tf.data pipeline)
│   └── utils/               # seed.py, losses.py, metrics.py, plots.py, gradcam.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Local setup

**1. Clone the repository**

```bash
git clone https://github.com/Deep-Learning-Models/ChestNet.git
cd ChestNet
```

**2. Create and activate a virtual environment**

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> 💡 **GPU:** on Linux or WSL2, install `tensorflow[and-cuda]` instead of `tensorflow` to train on an NVIDIA GPU. Native Windows runs TensorFlow on the CPU only, so use **WSL2** or **Google Colab** for GPU training.

---

## ▶️ How to run

Run every command from the repository root.

**1. Create the patient-level splits (once, shared by all models)**

```bash
python -m src.preprocessing.splits
```

This writes `data/splits/train.csv`, `val.csv`, `test.csv`, `summary.csv` (class counts per split) and `split_report.json` (proof that no patient or image is shared between splits).

**2. Explore the data (EDA)**

```bash
python -m src.eda
```

This saves report-ready figures and `eda_summary.json` in `results/eda/`: label distributions, co-occurrence, sample X-rays, images per patient, age and gender, AP/PA view share and class balance per split.

**3. Verify the shared training rules (before training any model)**

```bash
python -m src.check_pipeline
```

This checks that no model config changes a shared setting, that the splits have no leakage, that the input pipeline is correct and that augmentation touches the training set only. It saves `results/protocol/experimental_protocol.md` (a table for Report Section 5), `class_weights.csv` and `augmentation_preview.png`.

**4. Train a model**

```bash
python -m src.train --config configs/custom_cnn.yaml
python -m src.train --config configs/resnet50.yaml
python -m src.train --config configs/efficientnet.yaml
python -m src.train --config configs/vit.yaml
```

Add `--epochs 2` for a quick test run, or `--deterministic` for fully repeatable GPU results.

**5. Evaluate on the test set (once per model, at the very end)**

```bash
python -m src.evaluate --config configs/resnet50.yaml
```

**6. Grad-CAM heatmaps (CNN models)**

```bash
python -m src.explain --config configs/resnet50.yaml --num-images 8
```

**Where the outputs go**

| File | Created by | Contents |
|------|-----------|----------|
| `saved_models/<model>_best.weights.h5` | train | Best weights (highest validation AUC) |
| `results/<model>/config_used.yaml` | train | Exact settings used for the run |
| `results/<model>/history.csv`, `training_curves.png` | train | Loss and AUC per epoch |
| `results/<model>/training_summary.json` | train | Parameters, time per epoch, hardware, best epoch |
| `results/<model>/test_metrics.json`, `test_per_class_metrics.csv` | evaluate | Precision, recall, F1, ROC-AUC |
| `results/<model>/test_roc_curves.png`, `test_confusion_matrices.png` | evaluate | Plots for the report |
| `results/comparison.csv` | evaluate | One row per model, for the comparison table |
| `results/<model>/gradcam.png` | explain | Heatmaps over test X-rays |

---

## 📊 Dataset

| Item | Detail |
|------|--------|
| **Source** | [Random Sample of NIH Chest X-ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/sample) (Kaggle), a random sample of [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data), NIH Clinical Center |
| **Size** | 5,606 frontal chest X-rays (1024 × 1024 PNG), randomly sampled from the 112,120 images of the full dataset |
| **Labels** | 14 disease labels (multi-label) plus *No Finding*, in `sample_labels.csv` |
| **Our scope** | 7 disease classes plus *No Finding*. Every image of the 7 diseases is used; *No Finding* is capped at 1,500 images |

**How to get the data**

1. Download the dataset from the link above.
2. Put the images in `data/raw/` (any sub-folder) and the label file (`sample_labels.csv`) in `data/`.
3. `data/raw/` and the label file are listed in `.gitignore`, so they are **never** pushed to GitHub.

> 💡 The code also works with the full dataset: set `data.csv_path` in `configs/base.yaml` to `data/Data_Entry_2017.csv`.

Full step-by-step instructions are in [`data/README.md`](data/README.md).

---

## ⚙️ Shared experimental protocol

Every model follows the same rules so the comparison is fair. The rules live in `configs/base.yaml`, and `src/config.py` **locks** them: if a model config tries to change a shared setting, the run stops with an error.

| 🔒 Identical for all four models | ✏️ Allowed to differ per model (logged and justified) |
|---|---|
| Seed, classes, patient-level splits, image size (224 × 224), augmentation, max epochs, optimiser (AdamW), cosine LR schedule, weighted BCE loss, early stopping on validation AUC | Learning rate, weight decay, batch size (GPU memory), architecture settings |

- 🖼️ **Preprocessing:** grayscale PNG → 224 × 224 with anti-aliasing → cached once as uint8 → 3-channel float (0–255). Each model normalises its own input as its first layer (Custom CNN ÷255, ResNet50 caffe BGR mean subtraction, EfficientNet built-in, ViT [-1, 1]), so the pipeline is identical and evaluation can never use the wrong normalisation.
- 🔄 **Augmentation (training set only):** small rotation (±10°), shift and zoom (5%), brightness and contrast jitter. Horizontal flip is **off** because it puts the heart on the wrong side; it can be switched on for an ablation.
- 🧍 **Patient-level split:** 70 / 15 / 15 by patient ID. Many random patient splits are tried and the most label-balanced one is kept, so rare diseases appear in validation and test. `split_report.json` records 0 shared patients and 0 shared images.
- ⚖️ **Class imbalance:** weighted binary cross-entropy, with weights (#negatives / #positives) computed from the **training split only**.
- 🧪 **Training:** sigmoid outputs, AdamW, cosine decay, dropout, early stopping and best checkpoint on validation macro AUC.
- 🎲 **Reproducibility:** `seed: 42` fixes Python, NumPy and TensorFlow randomness. Every run saves its exact config, overrides, class weights, hardware and training time to `results/<model>/`.
- 🔒 **The test set is evaluated once**, at the very end.

---

## 📈 Evaluation

| Category | What we measure |
|----------|-----------------|
| 🎯 Classification quality | Precision, recall, F1 (macro and per class), ROC-AUC per disease, confusion matrices, accuracy (context only) |
| 📉 Learning behaviour | Training vs validation loss and accuracy curves (overfitting check) |
| ⏱️ Cost and generalisation | Parameter count, training time per epoch, inference time per image, validation-to-test gap |
| 🔍 Interpretability | Grad-CAM heatmaps for the CNN-based models |

### 🏆 Results

> Results will be added as each model is trained.

| Model | Macro F1 | Macro ROC-AUC | Params | Train time / epoch | Inference / image |
|-------|:--------:|:-------------:|:------:|:------------------:|:-----------------:|
| Custom CNN | – | – | – | – | – |
| ResNet50 | – | – | – | – | – |
| EfficientNet | – | – | – | – | – |
| ViT-B/16 | – | – | – | – | – |

---

## 👥 Team

| Member | Primary ownership |
|--------|-------------------|
| Member 1 | Dataset, EDA, preprocessing pipeline, Custom CNN |
| Member 2 | ResNet50 transfer-learning pipeline |
| Member 3 | EfficientNet and Grad-CAM interpretability |
| Member 4 | Vision Transformer and results comparison dashboard |

---

## 🌿 Contribution workflow

1. Create a feature branch from `main`, for example `feature/resnet50`.
2. Commit small, meaningful changes with clear messages.
3. Push and open a **pull request** into `main`.
4. Another team member reviews it before it is merged.

Every member commits and pushes **at least once a week**.

---

## 📄 License

Released under the [MIT License](LICENSE).
