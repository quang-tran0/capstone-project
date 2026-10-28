import numpy as np


class LayerNorm:
    """Normalize each token over its hidden dimension, using population variance."""

    def __init__(self, tensors, prefix="embeddings.LayerNorm", eps=1e-12):
        self.weight = tensors[f"{prefix}.weight"].astype(np.float32)
        self.bias = tensors[f"{prefix}.bias"].astype(np.float32)
        self.eps = np.float32(eps)

    def __call__(self, hidden_states):
        x = hidden_states.astype(np.float32)
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        variance = (centered * centered).mean(axis=-1, keepdims=True)
        normalized = centered / np.sqrt(variance + self.eps)
        return normalized * self.weight + self.bias
