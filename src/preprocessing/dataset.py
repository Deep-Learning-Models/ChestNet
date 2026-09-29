"""Step 6: tf.data input pipeline shared by all four models.

    PNG (1024x1024 grey) -> resize 224x224 (anti-aliased) -> uint8 cache
        -> [train only] shuffle -> batch -> [train only] augmentation -> 3-channel float32 (0-255)

Design decisions (explain these in Report Section 4)
----------------------------------------------------
* ONE pipeline for every model. It outputs float32 images in the 0-255 range and each
  model applies its OWN normalisation as the first layer inside the network:
      Custom CNN   -> Rescaling(1/255)               (0-1)
      ResNet50     -> RGB->BGR + ImageNet mean subtraction ("caffe" mode)
      EfficientNet -> built-in Rescaling + Normalization layers (expects 0-255)
      ViT-B/16     -> Rescaling to [-1, 1]
  Doing it inside the model means the saved model always receives raw pixels, so
  evaluation, Grad-CAM and the demo can never apply the wrong normalisation.
* Resized images are cached once as uint8 (4x smaller than float32). Decoding the
  1024x1024 PNGs is the slowest part, so every epoch after the first is much faster.
* Augmentation is applied to the TRAINING set only; validation and test images are
  never changed.
* Class weights are computed from the TRAINING split only (no information leaks from
  validation/test into training).
"""
import hashlib
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_image(path, image_size: int):
    """Read one X-ray -> (size, size, 1) uint8. Anti-aliasing avoids jagged edges when
    shrinking 1024 px images to 224 px (fine details such as nodules survive better)."""
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=1)                  # X-rays are grayscale
    img = tf.image.resize(img, (image_size, image_size), method="bilinear", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0.0, 255.0), tf.uint8)


def to_model_input(img):
    """uint8 grey (…, H, W, 1) -> float32 RGB (…, H, W, 3) in 0-255 (pretrained models need 3 channels)."""
    return tf.image.grayscale_to_rgb(tf.cast(img, tf.float32))


def _load_image(path, image_size: int):
    """Single image ready for a model: float32 RGB 0-255 (used by explain.py / demos)."""
    return to_model_input(load_image(path, image_size))


# --------------------------------------------------------------------------------------
# Augmentation (training only)
# --------------------------------------------------------------------------------------
def build_augmenter(aug_cfg: dict, seed: int) -> keras.Sequential:
    """Mild, clinically plausible augmentations only.

    * Small rotation / shift / zoom  -> patient positioning varies between scans.
    * Brightness / contrast jitter   -> exposure differs between X-ray machines.
    * Horizontal flip is OFF by default: it puts the heart on the right side
      (dextrocardia), an anatomy that almost never occurs, which may confuse
      side-dependent findings such as Cardiomegaly. It can be switched on in
      base.yaml for an ablation study.
    * No vertical flips, strong crops or colour changes (not realistic for X-rays).
    """
    layers = []
    if aug_cfg.get("horizontal_flip"):
        layers.append(keras.layers.RandomFlip("horizontal", seed=seed))
    if aug_cfg.get("rotation"):
        layers.append(keras.layers.RandomRotation(aug_cfg["rotation"], fill_mode="constant", seed=seed))
    if aug_cfg.get("translation"):
        t = aug_cfg["translation"]
        layers.append(keras.layers.RandomTranslation(t, t, fill_mode="constant", seed=seed))
    if aug_cfg.get("zoom"):
        layers.append(keras.layers.RandomZoom(aug_cfg["zoom"], fill_mode="constant", seed=seed))
    if aug_cfg.get("contrast"):
        layers.append(keras.layers.RandomContrast(aug_cfg["contrast"], value_range=(0, 255), seed=seed))
    if aug_cfg.get("brightness"):
        layers.append(keras.layers.RandomBrightness(aug_cfg["brightness"], value_range=(0, 255), seed=seed))
    return keras.Sequential(layers, name="augmentation")


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
def _cache_target(df, cfg: dict, split: str):
    """'' = in-memory cache, a file path = disk cache, None = no cache.

    The file name contains a hash of the image list + image size, so a new split or a
    new image size automatically creates a fresh cache (no stale data)."""
    mode = str(cfg["data"].get("cache", "disk")).lower()
    if mode in ("none", "false", "off"):
        return None
    if mode == "memory":
        return ""
    key = hashlib.md5(("|".join(df["image"].astype(str)) + str(cfg["data"]["image_size"])).encode()).hexdigest()[:10]
    cache_dir = Path(cfg["data"].get("cache_dir", "data/processed/cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / f"{split}_{cfg['data']['image_size']}px_{key}")


def make_dataset(df, cfg: dict, training: bool = False, split: str = None) -> tf.data.Dataset:
    """Build the batched tf.data pipeline for one split.

    training=True  -> shuffle + augmentation (train split only)
    training=False -> fixed order, no augmentation (validation / test)
    """
    classes = cfg["data"]["classes"]
    size = cfg["data"]["image_size"]
    batch = cfg["data"]["batch_size"]
    seed = cfg["seed"]
    split = split or ("train" if training else "eval")

    paths = df["path"].astype(str).to_numpy()
    labels = df[classes].to_numpy(dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (load_image(p, size), y), num_parallel_calls=AUTOTUNE)
    target = _cache_target(df, cfg, split)
    if target is not None:
        ds = ds.cache(target)                                   # decode + resize only once

    if training:
        ds = ds.shuffle(min(len(df), 10_000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch)
    ds = ds.map(lambda x, y: (to_model_input(x), y), num_parallel_calls=AUTOTUNE)

    if training:
        augmenter = build_augmenter(cfg.get("augmentation", {}), seed)
        if augmenter.layers:
            ds = ds.map(lambda x, y: (tf.clip_by_value(augmenter(x, training=True), 0.0, 255.0), y),
                        num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


# --------------------------------------------------------------------------------------
# Class imbalance (computed from the TRAINING split only)
# --------------------------------------------------------------------------------------
def compute_pos_weights(train_df, classes, max_weight: float = None) -> np.ndarray:
    """Weight for the positive samples of each class = #negatives / #positives.

    A rare disease (few positives) gets a larger weight, so missing it costs more in
    the loss. Always pass the TRAIN dataframe – never validation or test.
    `max_weight` optionally caps extreme weights for very rare classes.
    """
    pos = train_df[classes].sum().to_numpy(dtype=np.float32)
    neg = len(train_df) - pos
    w = neg / np.maximum(pos, 1.0)
    if max_weight:
        w = np.minimum(w, max_weight)
    return w
