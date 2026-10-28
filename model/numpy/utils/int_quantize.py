import numpy as np

from .fixed import FRAC_BITS, round_shift


# A scale is m * 2**-n with m in [1, 127] (INT8) and n in [0, 31] (UINT5).
# Per-channel weight scales go below 2**-15, so n needs five bits.
MULTIPLIER_MAX = 127
DEGREE_MAX = 31

# x / 127 as a multiplication by 1/127 in Q.24, rounded up so the scale
# never falls below peak / 127.
RECIP_127_BITS = 24
RECIP_127 = -(-(1 << RECIP_127_BITS) // MULTIPLIER_MAX)

# 1/m in Q.22 for m in [1, 127]; entry 0 is unused.
RECIP_M_BITS = 22
RECIP_M_LUT = np.array(
    [0] + [round((1 << RECIP_M_BITS) / m) for m in range(1, MULTIPLIER_MAX + 1)],
    dtype=np.int64,
)


def quantize_weight(weight, axis=0):
    """Offline: symmetric INT8 with one scale per slice, reducing over `axis`.

    Returns codes and the (m, n) scale fields, kept broadcastable to weight.
    """
    weight = np.asarray(weight, dtype=np.float32)
    scale = np.abs(weight).max(axis=axis, keepdims=True).astype(np.float64) / MULTIPLIER_MAX
    scale = np.maximum(scale, 2.0**-DEGREE_MAX)

    # Largest n whose multiplier still fits INT8. Rounding m up keeps the
    # peak at or below code 127, so nothing is clipped.
    degree = np.clip(np.floor(np.log2(MULTIPLIER_MAX / scale)), 0, DEGREE_MAX)
    degree = np.where(np.ceil(scale * 2.0**degree) > MULTIPLIER_MAX, degree - 1, degree)
    multiplier = np.maximum(np.ceil(scale * 2.0**degree), 1)

    step = (multiplier * 2.0**-degree).astype(np.float32)
    codes = np.clip(np.rint(weight / step), -MULTIPLIER_MAX, MULTIPLIER_MAX)
    return codes.astype(np.int8), multiplier.astype(np.int64), degree.astype(np.int64)


def scale_from_peak(peak, frac_bits=FRAC_BITS):
    """Scale fields (m, n) with m * 2**-n ~= peak * 2**-frac_bits / 127.

    The largest n whose multiplier fits INT8 is chosen, like a priority
    encoder over the 32 candidates. m is rounded up so the peak maps to a
    code <= 127. A zero peak gets the smallest scale.
    """
    peak = np.asarray(peak, dtype=np.int64)
    scaled = peak * RECIP_127

    # Ceiling shift: (v + 2**k - 1) >> k.
    shifts = frac_bits + RECIP_127_BITS - np.arange(DEGREE_MAX + 1)
    candidates = (scaled[..., None] + (np.int64(1) << shifts) - 1) >> shifts
    fits = candidates <= MULTIPLIER_MAX
    if not fits[..., 0].all():
        raise OverflowError("Peak too large for an INT8 multiplier at n = 0")

    degree = fits.sum(axis=-1) - 1
    multiplier = np.take_along_axis(candidates, degree[..., None], axis=-1)[..., 0]

    zero = multiplier == 0
    return np.where(zero, 1, multiplier), np.where(zero, DEGREE_MAX, degree)


def quantize_fixed(x, axis=-1, frac_bits=FRAC_BITS):
    """Symmetric INT8 of a Q.frac_bits tensor with one scale per slice along axis.

    codes = x * 2**-frac_bits / (m * 2**-n), with 1/m from RECIP_M_LUT.
    """
    x = np.asarray(x, dtype=np.int64)
    peak = np.abs(x).max(axis=axis, keepdims=True)
    multiplier, degree = scale_from_peak(peak, frac_bits)

    codes = round_shift(x * RECIP_M_LUT[multiplier], RECIP_M_BITS + frac_bits - degree)
    codes = np.clip(codes, -MULTIPLIER_MAX, MULTIPLIER_MAX)
    return codes.astype(np.int8), multiplier, degree


def dequantize(codes, multiplier, degree):
    """Debug/trace only: codes * m * 2**-n as float64."""
    return np.asarray(codes, dtype=np.float64) * multiplier * np.exp2(-np.asarray(degree, dtype=np.float64))
