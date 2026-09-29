"""tf.data input pipeline: load PNG -> 3-channel -> resize -> (augment) -> batch.

Images are returned as float32 in the 0-255 range. Each model applies its own
normalisation inside the network (see src/models), so one pipeline serves all four.
"""
import numpy as np
import tensorflow as tf
import keras

AUTOTUNE = tf.data.AUTOTUNE


def _load_image(path, image_size):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=1)          # X-rays are grayscale
    img = tf.image.grayscale_to_rgb(img)             # pretrained models expect 3 channels
    img = tf.image.resize(img, (image_size, image_size))
    return tf.cast(img, tf.float32)


def build_augmenter(aug_cfg: dict, seed: int) -> keras.Sequential:
    layers = []
    if aug_cfg.get("horizontal_flip"):
        layers.append(keras.layers.RandomFlip("horizontal", seed=seed))
    if aug_cfg.get("rotation"):
        layers.append(keras.layers.RandomRotation(aug_cfg["rotation"], fill_mode="constant", seed=seed))
    if aug_cfg.get("contrast"):
        layers.append(keras.layers.RandomContrast(aug_cfg["contrast"], value_range=(0, 255), seed=seed))
    if aug_cfg.get("brightness"):
        layers.append(keras.layers.RandomBrightness(aug_cfg["brightness"], value_range=(0, 255), seed=seed))
    return keras.Sequential(layers, name="augmentation")


def make_dataset(df, cfg: dict, training: bool = False) -> tf.data.Dataset:
    classes = cfg["data"]["classes"]
    size = cfg["data"]["image_size"]
    batch = cfg["data"]["batch_size"]

    paths = df["path"].astype(str).to_numpy()
    labels = df[classes].to_numpy(dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(df), seed=cfg["seed"], reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (_load_image(p, size), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch)
    if training:
        augmenter = build_augmenter(cfg.get("augmentation", {}), cfg["seed"])
        ds = ds.map(lambda x, y: (tf.clip_by_value(augmenter(x, training=True), 0.0, 255.0), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def compute_pos_weights(df, classes) -> np.ndarray:
    """Weight for positive samples of each class = (#negatives / #positives)."""
    pos = df[classes].sum().to_numpy(dtype=np.float32)
    neg = len(df) - pos
    return neg / np.maximum(pos, 1.0)
