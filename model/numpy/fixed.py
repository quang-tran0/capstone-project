import numpy as np


def signed_bits(bound):
    """Số bit có dấu cần để chứa mọi giá trị trong [-bound, bound]."""
    return int(bound).bit_length() + 1


def check_bits(x, bits, name):
    """Báo lỗi nếu x không vừa thanh ghi `bits` bit có dấu trên FPGA."""
    limit = 1 << (bits - 1)
    x = np.asarray(x)

    if x.size and (x.min() < -limit or x.max() >= limit):
        raise OverflowError(f"{name} vượt quá {bits} bit có dấu")

    return x


def bit_length(x):
    """Bộ mã hoá ưu tiên: số bit cần để biểu diễn x >= 0 (0 với x = 0)."""
    x = np.asarray(x, dtype=np.int64)
    length = np.zeros_like(x)

    for bit in range(63):
        length = np.where(x >> bit != 0, bit + 1, length)

    return length


def round_shift(x, shift):
    """Dịch phải có làm tròn (cộng nửa LSB rồi dịch số học); shift <= 0 là dịch trái."""
    x = np.asarray(x, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64)

    right = np.maximum(shift, 0)
    half = np.where(right > 0, np.int64(1) << np.maximum(right - 1, 0), 0)
    shifted = (x + half) >> right

    return shifted << np.maximum(-shift, 0)
