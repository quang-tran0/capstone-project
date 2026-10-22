import numpy as np

from ..utils.quantize import symmetric_dequantize, symmetric_quantize


class MeanPooling:
    """Masked integer sum followed by averaging and INT8/ScaleInt12 output."""

    def __call__(self, hidden_states, attention_mask=None):
        codes, scale = hidden_states
        codes = np.asarray(codes, dtype=np.int64)
        if codes.ndim != 3 or codes.shape[1] == 0:
            raise ValueError("Pooling expects non-empty codes with shape (B, N, H)")
        if attention_mask is None:
            attention_mask = np.ones(codes.shape[:2], dtype=np.int64)
        mask = np.asarray(attention_mask)
        if mask.shape != codes.shape[:2] or not np.all((mask == 0) | (mask == 1)):
            raise ValueError("Pooling mask must be binary with shape (B, N)")
        summed = (codes * mask.astype(np.int64)[..., None]).sum(axis=1)
        counts = mask.sum(axis=1, keepdims=True).clip(min=1)
        return symmetric_quantize(symmetric_dequantize(summed, scale) / counts)
