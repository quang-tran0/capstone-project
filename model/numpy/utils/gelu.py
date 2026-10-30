import math

import numpy as np

from ..hwtypes import Fixed


# GELU(x) = x * Phi(x); Phi is tabulated on [-8, 8) with step 2**-6.
PHI_STEP_BITS = 6
PHI_RANGE = 8
PHI_FRAC_BITS = 15


def prepare_phi_lut(step_bits=PHI_STEP_BITS, limit=PHI_RANGE, fraction_bits=PHI_FRAC_BITS):
    """Offline: ROM of the standard normal CDF Phi(x) in Q.fraction_bits."""
    steps = 1 << step_bits
    x = (np.arange(2 * limit * steps) - limit * steps) / steps
    phi = [0.5 * (1 + math.erf(value / math.sqrt(2))) for value in x]
    return Fixed(np.round(np.asarray(phi) * (1 << fraction_bits)).astype(np.int64),
                 fraction_bits + 1, frac=fraction_bits, signed=False)


def gelu_fixed(x, lut):
    """GELU of a Fixed x, returned in x's fraction."""
    # Address = round(x * 2**PHI_STEP_BITS) + offset, saturated to the table.
    offset = PHI_RANGE << PHI_STEP_BITS
    address = (x.round_shift(x.frac - PHI_STEP_BITS) + offset).clip(0, len(lut) - 1)
    return (x * lut[address]).round_shift(lut.frac)
