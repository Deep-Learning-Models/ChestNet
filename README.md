<p align="center">
  <img src="docs/assets/banner.png" alt="ChestNet banner" width="100%">
</p>

# 🫁 ChestNet

**Multi-Model Deep Learning Framework for Chest X-ray Multi-Disease Classification**

![Status](https://img.shields.io/badge/status-foundation%20ready-2ea44f)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/dataset-NIH%20ChestX--ray14-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**Current status:** 🚧 FOUNDATION READY. The repository structure, data rules and team workflow are in place, and model implementations are in progress.

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
| 4 | 🟪 **Vision Transformer (ViT-B/16)** | Attention-based | Image patches become tokens, and self-attention captures global context across both lungs. |

---

## 🛠️ Technology

- **Language:** Python 3.10+
- **Deep learning:** PyTorch, torchvision, timm (ViT and EfficientNet backbones)
- **Data and metrics:** NumPy, pandas, scikit-learn
- **Visualisation:** Matplotlib, Seaborn, Grad-CAM
- **Workflow:** Jupyter notebooks for EDA and experiments, `src/` for reusable code

---

## 📁 Project structure

```text
ChestNet/
├── data/              # Dataset instructions only (images are NOT committed)
├── docs/              # Diagrams, report assets and documentation
│   └── assets/
├── notebooks/         # EDA and one experiment notebook per model
├── results/           # Metrics, plots, confusion matrices, Grad-CAM outputs
├── src/
│   ├── models/        # custom_cnn.py, resnet50.py, efficientnet.py, vit.py
│   ├── preprocessing/ # Loading, resizing, augmentation, patient-level splits
│   └── utils/         # Metrics, Grad-CAM, seeding, plotting helpers
├── requirements.txt
├── .gitignore
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

> 💡 For GPU training, install the CUDA build of PyTorch that matches your system from [pytorch.org](https://pytorch.org/get-started/locally/) **before** running the command above.

---

## 📊 Dataset

| Item | Detail |
|------|--------|
| **Source** | [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data), NIH Clinical Center |
| **Size** | 112,120 frontal chest X-rays from 30,805 patients |
| **Labels** | 14 disease labels (multi-label) plus *No Finding* |
| **Our scope** | 5–8 disease classes plus *No Finding*, with a balanced subset of about 1,000–2,000 images per class |

**How to get the data**

1. Download the dataset from the link above.
2. Put the images in `data/raw/` and the label file (`Data_Entry_2017.csv`) in `data/`.
3. `data/raw/` and `data/processed/` are listed in `.gitignore`, so images are **never** pushed to GitHub.

---

## ⚙️ Shared experimental protocol

Every model follows the same rules so the comparison is fair.

- 🖼️ **Preprocessing:** resize to 224 × 224 and normalise with ImageNet mean and standard deviation.
- 🔄 **Augmentation:** small rotations, horizontal flips, and brightness/contrast jitter. Only clinically plausible changes are used.
- 🧍 **Patient-level split:** each patient's images stay in **one** split (train / validation / test), which prevents data leakage.
- ⚖️ **Class imbalance:** weighted binary cross-entropy loss.
- 🧪 **Training:** sigmoid outputs, Adam/AdamW, cosine learning-rate decay, dropout and early stopping.
- 🎲 **Reproducibility:** fixed random seeds, saved configs, and logged hardware and training time.
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
