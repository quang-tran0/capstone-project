import numpy as np


# Every wide tensor (residual stream, Linear outputs, LayerNorm) is Q.16 INT32.
FRAC_BITS = 16
WIDE_BITS = 32


def to_float(x, frac_bits=FRAC_BITS):
    """Debug/trace only: raw Q.frac_bits integers to float64."""
    return np.asarray(x, dtype=np.float64) / (1 << frac_bits)
