import numpy as np


def check_bits(x, bits, name):
    """Raise if x does not fit a signed register of the given width."""
    limit = 1 << (bits - 1)
    x = np.asarray(x)

    if x.size and (x.min() < -limit or x.max() >= limit):
        raise OverflowError(f"{name} exceeds {bits} signed bits")

    return x


def bit_length(x):
    """Priority encoder: bits needed to represent x >= 0 (0 for x = 0)."""
    x = np.asarray(x, dtype=np.int64)
    length = np.zeros_like(x)

    for bit in range(63):
        length = np.where(x >> bit != 0, bit + 1, length)

    return length


def round_shift(x, shift):
    """Right shift rounding half up; a negative shift is a left shift."""
    x = np.asarray(x, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64)

    right = np.maximum(shift, 0)
    half = np.where(right > 0, np.int64(1) << np.maximum(right - 1, 0), 0)
    shifted = (x + half) >> right

    return shifted << np.maximum(-shift, 0)
