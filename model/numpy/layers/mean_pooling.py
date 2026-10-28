import numpy as np

from ..utils.fixed import round_shift
from ..utils.reciprocal import prepare_reciprocal_lut, reciprocal_fixed


class MeanPooling:
    """Masked integer sum, then a multiply by 1/count; Q.16 in and out."""

    def __init__(self):
        self.reciprocal_lut = prepare_reciprocal_lut()

    def __call__(self, hidden_states, attention_mask=None):
        hidden_states = np.asarray(hidden_states, dtype=np.int64)
        if hidden_states.ndim != 3 or hidden_states.shape[1] == 0:
            raise ValueError("Pooling expects a non-empty tensor with shape (B, N, H)")
        if attention_mask is None:
            attention_mask = np.ones(hidden_states.shape[:2], dtype=np.int64)
        mask = np.asarray(attention_mask)
        if mask.shape != hidden_states.shape[:2] or not np.all((mask == 0) | (mask == 1)):
            raise ValueError("Pooling mask must be binary with shape (B, N)")
        summed = (hidden_states * mask.astype(np.int64)[..., None]).sum(axis=1)
        counts = mask.sum(axis=1, keepdims=True).clip(min=1)
        mantissa, shift = reciprocal_fixed(counts, self.reciprocal_lut)
        return round_shift(summed * mantissa, shift).astype(np.int32)
