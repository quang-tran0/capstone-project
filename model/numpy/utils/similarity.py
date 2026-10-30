import numpy as np

from ..hwtypes import Fixed, Scale, leading_one
from .fixed import FRAC_BITS
from .rsqrt import prepare_rsqrt_lut, rsqrt_fixed


# Vectors are reduced to 16 signed bits before the dot product.
VECTOR_BITS = 16
SCORE_FRAC_BITS = 15
SCORE_BITS = 17


def _reduce(x):
    """Per-vector dynamic shift so every element fits VECTOR_BITS; the scale cancels."""
    peak = abs(x).max(axis=-1, keepdims=True)
    shift = (leading_one(peak) + (1 - VECTOR_BITS)).clip(0)
    return (x >> shift).with_frac(0).resize(VECTOR_BITS)


def cosine_fixed(a, b, lut=None):
    """Cosine similarity over the last axis in Q.15 (raw np.int32).

    a and b are Q.16 sentence embeddings, as raw arrays or Fixed.
    """
    lut = prepare_rsqrt_lut() if lut is None else lut
    a, b = (x if isinstance(x, Fixed) else Fixed(x, 32, FRAC_BITS) for x in (a, b))
    a, b = _reduce(a), _reduce(b)
    shape = np.broadcast_shapes(a.shape, b.shape)
    a, b = a.broadcast_to(shape), b.broadcast_to(shape)

    dot = (a * b).sum(axis=-1)
    inverse_a = rsqrt_fixed((a * a).sum(axis=-1).clip(1), lut)
    inverse_b = rsqrt_fixed((b * b).sum(axis=-1).clip(1), lut)

    # dot / |a| first so the intermediate stays inside 64 bits.
    partial = inverse_a.rescale(dot, frac=SCORE_FRAC_BITS)
    score = inverse_b.rescale(partial, frac=SCORE_FRAC_BITS)
    return score.resize(SCORE_BITS, name="cosine score").raw.astype(np.int32)
