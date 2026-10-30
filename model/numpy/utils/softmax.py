import math

import numpy as np

from ..hwtypes import Fixed, where


# log2(e) in Q.15: exp(-t) = 2**(-t * log2(e)).
LOG2E_BITS = 15
LOG2E = Fixed.const(round(math.log2(math.e) * (1 << LOG2E_BITS)), frac=LOG2E_BITS)

# 2**-f LUT for f in [0, 1] with step 2**-8; values are Q.15, LUT[0] = 1.0.
EXP2_INDEX_BITS = 8
EXP2_FRAC_BITS = 15

# Probabilities relative to the row maximum, 1.0 -> 127.
PROBABILITY_MAX = 127
PROBABILITY_BITS = 8


def prepare_exp2_lut(index_bits=EXP2_INDEX_BITS, fraction_bits=EXP2_FRAC_BITS):
    """Offline: ROM of 2**(-k / 2**index_bits) for k = 0 .. 2**index_bits, Q.fraction_bits."""
    steps = 1 << index_bits
    values = np.round((1 << fraction_bits) * np.exp2(-np.arange(steps + 1) / steps))
    return Fixed(values.astype(np.int64), fraction_bits + 1, frac=fraction_bits, signed=False)


def exp_neg_fixed(t, lut):
    """exp(-t) in Q.15 for a Fixed t >= 0."""
    # w = t * log2(e) = integer part + fraction part, in t's format.
    w = (t * LOG2E).round_shift(LOG2E_BITS)
    integer = w.floor_shift(w.frac)
    fraction = w & ((1 << w.frac) - 1)

    # 2**-w = LUT(fraction) >> integer.
    value = lut[fraction.round_shift(w.frac - EXP2_INDEX_BITS)]
    return where(integer < EXP2_FRAC_BITS + 1, value >> integer.clip(0, 63), 0)


def softmax_fixed(z, mask, lut):
    """Integer softmax over the last axis of a Fixed z.

    Returns INT8 codes in [0, 127] and the row total E (Q.15), with
    softmax(z)_j ~= codes_j * 2**15 / (127 * E).
    """
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), z.shape)

    # Subtract the row maximum so every exponent is <= 0; masked keys are skipped.
    peak = z.max(axis=-1, keepdims=True, where=mask)
    exponent = exp_neg_fixed(where(mask, peak - z, 0), lut)
    exponent = where(mask, exponent, 0)

    total = exponent.sum(axis=-1, keepdims=True)
    codes = (exponent * PROBABILITY_MAX).round_shift(EXP2_FRAC_BITS)
    return codes.resize(PROBABILITY_BITS, signed=True), total
