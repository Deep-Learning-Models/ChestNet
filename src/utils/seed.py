"""Fix every source of randomness so experiments can be reproduced."""
import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic_ops: bool = False) -> None:
    """Seed Python, NumPy and TensorFlow/Keras.

    deterministic_ops=True also forces deterministic GPU kernels
    (fully repeatable results, but training can be slower).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(seed)  # seeds python, numpy and tf together
    tf.random.set_seed(seed)
    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()
