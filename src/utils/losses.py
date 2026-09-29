"""Weighted binary cross-entropy for imbalanced multi-label classification."""
import tensorflow as tf


def weighted_bce(pos_weights):
    """Positive examples of rare diseases count more (weight = #neg / #pos per class)."""
    w = tf.constant(pos_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        per_label = -(w * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(per_label, axis=-1)

    loss.__name__ = "weighted_bce"
    return loss
