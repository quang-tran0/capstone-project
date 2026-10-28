import math

import numpy as np

from .fixed import FRAC_BITS, round_shift


# GELU(x) = x * Phi(x); Phi is tabulated on [-8, 8) with step 2**-6.
PHI_STEP_BITS = 6
PHI_RANGE = 8
PHI_FRAC_BITS = 15


def prepare_phi_lut(step_bits=PHI_STEP_BITS, limit=PHI_RANGE, fraction_bits=PHI_FRAC_BITS):
    """Offline: the standard normal CDF Phi(x) in Q.fraction_bits."""
    steps = 1 << step_bits
    x = (np.arange(2 * limit * steps) - limit * steps) / steps
    phi = [0.5 * (1 + math.erf(value / math.sqrt(2))) for value in x]
    return np.round(np.asarray(phi) * (1 << fraction_bits)).astype(np.int32)


def gelu_fixed(x, lut, frac_bits=FRAC_BITS):
    """GELU of a Q.frac_bits tensor, returned in the same format."""
    x = np.asarray(x, dtype=np.int64)

    # Address = round(x * 2**PHI_STEP_BITS) + offset, saturated to the table.
    offset = PHI_RANGE << PHI_STEP_BITS
    index = np.clip(round_shift(x, frac_bits - PHI_STEP_BITS) + offset, 0, lut.size - 1)

    return round_shift(x * lut[index], PHI_FRAC_BITS)
