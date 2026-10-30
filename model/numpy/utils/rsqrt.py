import numpy as np

from ..hwtypes import Fixed, Scale, leading_one


# 1/sqrt(m) LUT for m in [1, 4): 3 * 2**RSQRT_INDEX_BITS entries.
RSQRT_INDEX_BITS = 8
RSQRT_FRAC_BITS = 16
# Mantissa after the Newton step, about 2**16 / sqrt(m) <= 2**16.
RSQRT_MANTISSA_BITS = 17


def prepare_rsqrt_lut(index_bits=RSQRT_INDEX_BITS, fraction_bits=RSQRT_FRAC_BITS):
    """Offline: ROM of 1/sqrt(m) sampled at the center of each m interval."""
    steps = 1 << index_bits
    centers = 1 + (np.arange(3 * steps) + 0.5) / steps
    return Fixed(np.round((1 << fraction_bits) / np.sqrt(centers)).astype(np.int64),
                 fraction_bits + 1, signed=False)


def rsqrt_fixed(value, lut, index_bits=RSQRT_INDEX_BITS, fraction_bits=RSQRT_FRAC_BITS):
    """1/sqrt of a positive Fixed (real value, its frac honoured) as a Scale m * 2**-n.

    The LUT gives a first guess, refined by one Newton step. value.frac must be even.
    """
    if value.size == 0 or (value.raw <= 0).any():
        raise ValueError("rsqrt requires positive integers")
    if value.frac % 2:
        raise ValueError("rsqrt needs an even fraction")
    # 1/sqrt(raw * 2**-f) = 2**(f/2) / sqrt(raw).
    half_frac = value.frac // 2
    value = value.with_frac(0)

    # value = m * 2**exponent with an even exponent and m in [1, 4) as Q.fraction_bits.
    exponent = ((leading_one(value) - 1) >> 1) << 1
    m = value.floor_shift(exponent - fraction_bits, frac=0).resize(fraction_bits + 2, signed=False)

    # The top bits of m address the LUT.
    y = lut[(m >> (fraction_bits - index_bits)) - (1 << index_bits)]

    # Newton: y = y * (3 - m * y**2) / 2.
    y_squared = (y * y) >> fraction_bits
    error = (m * y_squared) >> fraction_bits
    y = (y * ((3 << fraction_bits) - error)) >> (fraction_bits + 1)

    return Scale(y.resize(RSQRT_MANTISSA_BITS, signed=False, name="rsqrt mantissa"),
                 (exponent >> 1) + (fraction_bits - half_frac))
