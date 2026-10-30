import numpy as np

from ..hwtypes import Fixed
from ..utils.fixed import WIDE_BITS
from ..utils.reciprocal import prepare_reciprocal_lut, reciprocal_fixed


class MeanPooling:
    """Masked integer sum, then a multiply by 1/count; Q.16 in and out."""

    def __init__(self):
        self.reciprocal_lut = prepare_reciprocal_lut()

    def __call__(self, hidden_states, attention_mask=None):
        if not isinstance(hidden_states, Fixed):
            raise TypeError("Pooling input must be a Fixed tensor")
        if hidden_states.ndim != 3 or hidden_states.shape[1] == 0:
            raise ValueError("Pooling expects a non-empty tensor with shape (B, N, H)")
        if attention_mask is None:
            attention_mask = np.ones(hidden_states.shape[:2], dtype=np.int64)
        mask = np.asarray(attention_mask)
        if mask.shape != hidden_states.shape[:2] or not np.all((mask == 0) | (mask == 1)):
            raise ValueError("Pooling mask must be binary with shape (B, N)")
        summed = (hidden_states * mask[..., None]).sum(axis=1)
        counts = Fixed.const(mask.sum(axis=1, keepdims=True).clip(min=1))
        inverse_count = reciprocal_fixed(counts, self.reciprocal_lut)
        return inverse_count.rescale(summed, hidden_states.frac).resize(WIDE_BITS, name="pooling")
