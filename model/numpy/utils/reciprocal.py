import numpy as np

from .fixed import bit_length


# 1/m LUT for m in [1, 2): 2**RECIP_INDEX_BITS entries.
RECIP_INDEX_BITS = 8
RECIP_FRAC_BITS = 16


def prepare_reciprocal_lut(index_bits=RECIP_INDEX_BITS, fraction_bits=RECIP_FRAC_BITS):
    """Offline: 1/m sampled at the center of each m interval."""
    steps = 1 << index_bits
    centers = 1 + (np.arange(steps) + 0.5) / steps

    return np.round((1 << fraction_bits) / centers).astype(np.int32)


def reciprocal_fixed(value, lut, index_bits=RECIP_INDEX_BITS, fraction_bits=RECIP_FRAC_BITS):
    """1/value ~= mantissa * 2**-shift, from the LUT plus one Newton step."""
    value = np.asarray(value, dtype=np.int64)

    if value.size == 0 or value.min() <= 0:
        raise ValueError("reciprocal requires positive integers")

    # value = m * 2**exponent with m in [1, 2) as Q.fraction_bits.
    exponent = bit_length(value) - 1
    shift = exponent - fraction_bits
    m = np.where(
        shift >= 0,
        value >> np.maximum(shift, 0),
        value << np.maximum(-shift, 0),
    )

    # The bits below the leading one address the LUT.
    index = (m >> (fraction_bits - index_bits)) - (1 << index_bits)
    y = lut[index].astype(np.int64)

    # Newton: y = y * (2 - m * y).
    error = (m * y) >> fraction_bits
    y = (y * ((2 << fraction_bits) - error)) >> fraction_bits

    return y, fraction_bits + exponent
