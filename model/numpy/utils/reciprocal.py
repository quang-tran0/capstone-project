import numpy as np

from ..hwtypes import Fixed, Scale, leading_one


# 1/m LUT for m in [1, 2): 2**RECIP_INDEX_BITS entries.
RECIP_INDEX_BITS = 8
RECIP_FRAC_BITS = 16
# Mantissa after the Newton step, about 2**16 / m <= 2**16.
RECIP_MANTISSA_BITS = 17


def prepare_reciprocal_lut(index_bits=RECIP_INDEX_BITS, fraction_bits=RECIP_FRAC_BITS):
    """Offline: ROM of 1/m sampled at the center of each m interval."""
    steps = 1 << index_bits
    centers = 1 + (np.arange(steps) + 0.5) / steps
    return Fixed(np.round((1 << fraction_bits) / centers).astype(np.int64),
                 fraction_bits + 1, signed=False)


def reciprocal_fixed(value, lut, index_bits=RECIP_INDEX_BITS, fraction_bits=RECIP_FRAC_BITS):
    """1/value of a positive Fixed (real value, its frac honoured) as a Scale m * 2**-n.

    The LUT gives a first guess, refined by one Newton step.
    """
    if value.size == 0 or (value.raw <= 0).any():
        raise ValueError("reciprocal requires positive integers")
    # 1/(raw * 2**-f) = 2**f / raw.
    value_frac = value.frac
    value = value.with_frac(0)

    # value = m * 2**exponent with m in [1, 2) as Q.fraction_bits.
    exponent = leading_one(value) - 1
    m = value.floor_shift(exponent - fraction_bits, frac=0).resize(fraction_bits + 1, signed=False)

    # The bits below the leading one address the LUT.
    y = lut[(m >> (fraction_bits - index_bits)) - (1 << index_bits)]

    # Newton: y = y * (2 - m * y).
    error = (m * y) >> fraction_bits
    y = (y * ((2 << fraction_bits) - error)) >> fraction_bits

    return Scale(y.resize(RECIP_MANTISSA_BITS, signed=False, name="reciprocal mantissa"),
                 exponent + (fraction_bits - value_frac))
