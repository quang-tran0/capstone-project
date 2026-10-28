import numpy as np


def softmax(x, axis=-1):
    """Stable float32 softmax for the NumPy reference model."""
    x = np.asarray(x, dtype=np.float32)
    exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp / np.sum(exp, axis=axis, keepdims=True)
