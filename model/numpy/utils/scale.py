import math

import numpy as np



def quantize_scale(scale: float) -> tuple[np.int8, np.uint8]:
    """Represent a scale in [0, 1] as multiplier * 2**(-n).

    The multiplier is signed INT8 (0..127 for a scale). The exponent n is
    unsigned 4-bit (0..15), stored in np.uint8 because NumPy has no UINT4
    dtype. Choose the largest n whose rounded multiplier fits in INT8.
    The finest step is 2**(-15); smaller values may round to zero.
    """
    scale = float(scale)
    if not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError("Scale must be finite and in [0, 1]")

    for n in range(15, -1, -1):
        multiplier = round(scale * 2**n)
        if multiplier <= 127:
            return np.int8(multiplier), np.uint8(n)

    raise ValueError("Scale cannot be represented as INT8 with a UINT4 exponent")


if __name__ == "__main__":
    for value, expected in (
        (0.0, (0, 15)),
        (2**-16, (0, 15)),
        (2**-15, (1, 15)),
        (1 / 256, (64, 14)),
        (1 / 128, (64, 13)),
        (0.5, (64, 7)),
        (127 / 128, (127, 7)),
        (1.0, (64, 6)),
        (127 / 256, (127, 8)),
    ):
        multiplier, degree = quantize_scale(value)
        assert (int(multiplier), int(degree)) == expected
        assert isinstance(multiplier, np.int8) and isinstance(degree, np.uint8)
        assert abs(value - int(multiplier) * 2.0**(-int(degree))) <= 2.0**(-int(degree) - 1)

    for value in (-1.0, 1.01, math.inf, math.nan):
        try:
            quantize_scale(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {value}")
