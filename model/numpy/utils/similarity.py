import numpy as np

from .fixed import bit_length, round_shift
from .rsqrt import prepare_rsqrt_lut, rsqrt_fixed


# Vectors are reduced to 16 signed bits before the dot product.
VECTOR_BITS = 16
SCORE_FRAC_BITS = 15


def _reduce(x):
    """Per-vector dynamic shift so every element fits VECTOR_BITS."""
    x = np.asarray(x, dtype=np.int64)
    peak = np.abs(x).max(axis=-1, keepdims=True)
    return x >> np.maximum(bit_length(peak) + 1 - VECTOR_BITS, 0)


def cosine_fixed(a, b, lut=None):
    """Cosine similarity of the last axis in Q.15; scales cancel."""
    lut = prepare_rsqrt_lut() if lut is None else lut
    a, b = np.broadcast_arrays(_reduce(a), _reduce(b))

    dot = (a * b).sum(axis=-1)
    a_mantissa, a_shift = rsqrt_fixed(np.maximum((a * a).sum(axis=-1), 1), lut)
    b_mantissa, b_shift = rsqrt_fixed(np.maximum((b * b).sum(axis=-1), 1), lut)

    # dot / |a| first so the intermediate stays inside 64 bits.
    partial = round_shift(dot * a_mantissa, a_shift - SCORE_FRAC_BITS)
    return round_shift(partial * b_mantissa, b_shift).astype(np.int32)
