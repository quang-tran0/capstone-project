import math

import numpy as np

from .fixed import FRAC_BITS, round_shift


# log2(e) in Q.15: exp(-t) = 2**(-t * log2(e)).
LOG2E_BITS = 15
LOG2E = round(math.log2(math.e) * (1 << LOG2E_BITS))

# 2**-f LUT for f in [0, 1] with step 2**-8; values are Q.15, LUT[0] = 1.0.
EXP2_INDEX_BITS = 8
EXP2_FRAC_BITS = 15

# Probabilities relative to the row maximum, 1.0 -> 127.
PROBABILITY_MAX = 127


def prepare_exp2_lut(index_bits=EXP2_INDEX_BITS, fraction_bits=EXP2_FRAC_BITS):
    """Offline: 2**(-k / 2**index_bits) for k = 0 .. 2**index_bits."""
    steps = 1 << index_bits
    return np.round((1 << fraction_bits) * np.exp2(-np.arange(steps + 1) / steps)).astype(np.int32)


def exp_neg_fixed(t, lut, frac_bits=FRAC_BITS):
    """exp(-t) in Q.15 for t >= 0 in Q.frac_bits."""
    # w = t * log2(e) = integer part + fraction part.
    w = round_shift(np.asarray(t, dtype=np.int64) * LOG2E, LOG2E_BITS)
    integer = w >> frac_bits
    fraction = w & ((1 << frac_bits) - 1)

    # 2**-w = LUT(fraction) >> integer.
    index = round_shift(fraction, frac_bits - EXP2_INDEX_BITS)
    value = lut[index].astype(np.int64)
    return np.where(integer < EXP2_FRAC_BITS + 1, value >> np.minimum(integer, 63), 0)


def softmax_fixed(z, mask, lut, frac_bits=FRAC_BITS):
    """Integer softmax over the last axis of z (Q.frac_bits).

    Returns codes in [0, 127] and the row total E (Q.15), with
    softmax(z)_j ~= codes_j * 2**15 / (127 * E).
    """
    z = np.asarray(z, dtype=np.int64)
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), z.shape)

    # Subtract the row maximum so every exponent is <= 0; masked keys are skipped.
    low = np.iinfo(np.int64).min
    peak = np.where(mask, z, low).max(axis=-1, keepdims=True)
    exponent = exp_neg_fixed(np.where(mask, peak - z, 0), lut, frac_bits)
    exponent = np.where(mask, exponent, 0)

    total = exponent.sum(axis=-1, keepdims=True)
    codes = round_shift(exponent * PROBABILITY_MAX, EXP2_FRAC_BITS)
    return codes.astype(np.int8), total
