import numpy as np

from .fixed import bit_length


# 1/sqrt(m) LUT for m in [1, 4): 3 * 2**RSQRT_INDEX_BITS entries.
RSQRT_INDEX_BITS = 8
RSQRT_FRAC_BITS = 16


def prepare_rsqrt_lut(index_bits=RSQRT_INDEX_BITS, fraction_bits=RSQRT_FRAC_BITS):
    """Offline: 1/sqrt(m) sampled at the center of each m interval."""
    steps = 1 << index_bits
    centers = 1 + (np.arange(3 * steps) + 0.5) / steps

    return np.round((1 << fraction_bits) / np.sqrt(centers)).astype(np.int32)


def rsqrt_fixed(value, lut, index_bits=RSQRT_INDEX_BITS, fraction_bits=RSQRT_FRAC_BITS):
    """1/sqrt(value) ~= mantissa * 2**-shift, from the LUT plus one Newton step."""
    value = np.asarray(value, dtype=np.int64)

    if value.size == 0 or value.min() <= 0:
        raise ValueError("rsqrt requires positive integers")

    # value = m * 2**exponent with an even exponent and m in [1, 4) as Q.fraction_bits.
    exponent = (bit_length(value) - 1) & ~1
    shift = exponent - fraction_bits
    m = np.where(
        shift >= 0,
        value >> np.maximum(shift, 0),
        value << np.maximum(-shift, 0),
    )

    # The top bits of m address the LUT.
    index = (m >> (fraction_bits - index_bits)) - (1 << index_bits)
    y = lut[index].astype(np.int64)

    # Newton: y = y * (3 - m * y**2) / 2.
    y_squared = (y * y) >> fraction_bits
    error = (m * y_squared) >> fraction_bits
    y = (y * ((3 << fraction_bits) - error)) >> (fraction_bits + 1)

    return y, fraction_bits + exponent // 2
