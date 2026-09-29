"""Grad-CAM: highlight the image regions that drove a CNN's prediction."""
import numpy as np
import tensorflow as tf
import keras


def gradcam_heatmap(model: keras.Model, image: np.ndarray, layer_name: str, class_index: int) -> np.ndarray:
    """Return a heatmap in [0, 1] with the same height/width as `image` (H, W, 3, 0-255)."""
    grad_model = keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    x = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        score = preds[:, class_index]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))                          # importance of each feature map
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights[0], axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], image.shape[:2])[..., 0]
    return cam.numpy()
