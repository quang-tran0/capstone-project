from functools import lru_cache

import numpy as np

from ..hwtypes import Fixed, QTensor, ScaleFormat


# A scale is m * 2**-n with m in [1, 127] (7 bits) and n in [0, 31] (5 bits).
# Per-channel weight scales go below 2**-15, so n needs five bits.
SCALE_FORMAT = ScaleFormat(m_bits=7, n_bits=5)
CODE_BITS = 8

# 1/m in Q.22 for m in [1, m_max]; entry 0 is unused.
RECIP_M_BITS = 22


def code_max(code_bits=CODE_BITS):
    """Symmetric codes use [-code_max, code_max], e.g. 127 for INT8."""
    return (1 << (code_bits - 1)) - 1


@lru_cache
def reciprocal_m_lut(m_bits=SCALE_FORMAT.m_bits):
    """Offline: ROM of 1/m in Q.RECIP_M_BITS, addressed by m."""
    values = [0] + [round((1 << RECIP_M_BITS) / m) for m in range(1, 1 << m_bits)]
    return Fixed(values, RECIP_M_BITS + 1, signed=False)


def quantize_weight(weight, axis=0, scale_format=SCALE_FORMAT, code_bits=CODE_BITS):
    """Offline: symmetric codes with one scale per slice, reducing over `axis`.

    The scale fields stay broadcastable to the weight. Rounding m up keeps the
    peak at or below code_max, so nothing is clipped.
    """
    limit = code_max(code_bits)
    weight = np.asarray(weight, dtype=np.float32)
    peak = np.abs(weight).max(axis=axis, keepdims=True).astype(np.float64)
    scale = scale_format.from_float(peak / limit)

    step = scale.to_float().astype(np.float32)
    codes = np.clip(np.rint(weight / step), -limit, limit).astype(np.int64)
    return QTensor(Fixed(codes, code_bits), scale)


def quantize(x, axis=-1, scale_format=SCALE_FORMAT, code_bits=CODE_BITS):
    """Runtime, integer-only: symmetric codes of a Fixed with one scale per slice along axis.

    codes = x / (m * 2**-n), using the 1/m ROM instead of a divider.
    """
    limit = code_max(code_bits)
    scale = scale_format.from_peak(abs(x).max(axis=axis, keepdims=True), limit)
    recip = reciprocal_m_lut(scale_format.m_bits)[scale.m]
    codes = (x * recip).round_shift(RECIP_M_BITS + x.frac - scale.n, frac=0)
    return QTensor(codes.clip(-limit, limit).resize(code_bits), scale)
