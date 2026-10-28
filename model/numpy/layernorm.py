import math

import numpy as np

if __package__:
    from .fixed import bit_length, check_bits, round_shift
else:
    from fixed import bit_length, check_bits, round_shift


# e = d >> shift vừa cổng 18 bit của DSP48E2 trước khi bình phương.
SQUARE_BITS = 18
# Bộ cộng dồn tổng bình phương, bằng độ rộng accumulator của DSP48E2.
ACCUMULATOR_BITS = 48

# LUT 1/sqrt(m) với m trong [1, 4): 3 * 2^RSQRT_INDEX_BITS ô.
RSQRT_INDEX_BITS = 8
RSQRT_FRAC_BITS = 16

# (x - mean) / ||x - mean|| nằm trong [-1, 1].
NORM_FRAC_BITS = 17
GAMMA_FRAC_BITS = 12
LAYERNORM_OUT_FRAC_BITS = 16


def prepare_rsqrt_lut(index_bits=RSQRT_INDEX_BITS, fraction_bits=RSQRT_FRAC_BITS):
    """Chạy trên máy tính: bảng 1/sqrt(m) lấy tại tâm mỗi khoảng của m."""
    steps = 1 << index_bits
    centers = 1 + (np.arange(3 * steps) + 0.5) / steps

    return np.round((1 << fraction_bits) / np.sqrt(centers)).astype(np.int32)


def prepare_layernorm(tensors, prefix):
    """Chạy trên máy tính: lượng tử hoá gamma và beta."""
    gamma = tensors[f"{prefix}.weight"].astype(np.float64)
    beta = tensors[f"{prefix}.bias"].astype(np.float64)
    size = gamma.shape[0]

    # Gộp sqrt(N) vào gamma để FPGA chỉ cần tính d / ||d||.
    gamma_fixed = np.round(gamma * math.sqrt(size) * (1 << GAMMA_FRAC_BITS))
    gamma_fixed = check_bits(gamma_fixed.astype(np.int32), SQUARE_BITS, "gamma")
    beta_fixed = np.round(beta * (1 << LAYERNORM_OUT_FRAC_BITS)).astype(np.int32)

    return gamma_fixed, beta_fixed


def rsqrt_fixed(value, lut, index_bits=RSQRT_INDEX_BITS, fraction_bits=RSQRT_FRAC_BITS):
    """1/sqrt(value) ~= mantissa * 2^-shift, dùng LUT và một bước Newton."""
    value = np.asarray(value, dtype=np.int64)

    if value.size == 0 or value.min() <= 0:
        raise ValueError("rsqrt chỉ nhận số nguyên dương")

    # value = m * 2^exponent với exponent chẵn, m trong [1, 4) dạng Q.fraction_bits.
    exponent = (bit_length(value) - 1) & ~1
    shift = exponent - fraction_bits
    m = np.where(
        shift >= 0,
        value >> np.maximum(shift, 0),
        value << np.maximum(-shift, 0),
    )

    # Bit cao của m là địa chỉ LUT.
    index = (m >> (fraction_bits - index_bits)) - (1 << index_bits)
    y = lut[index].astype(np.int64)

    # Newton: y = y * (3 - m * y^2) / 2.
    y_squared = (y * y) >> fraction_bits
    error = (m * y_squared) >> fraction_bits
    y = (y * ((3 << fraction_bits) - error)) >> (fraction_bits + 1)

    return y, fraction_bits + exponent // 2


def layernorm_fixed(x, gamma_fixed, beta_fixed, lut):
    """LayerNorm số nguyên theo trục cuối, kết quả Q.LAYERNORM_OUT_FRAC_BITS.

    Định dạng Q của x không ảnh hưởng kết quả vì d / ||d|| không phụ thuộc thang đo.
    """
    x = np.asarray(x, dtype=np.int64)
    size = x.shape[-1]

    # d = N * (x - mean): không cần phép chia cho N.
    total = x.sum(axis=-1, keepdims=True)
    deviation = size * x - total

    # Dịch động theo từng vector để e dùng đủ SQUARE_BITS bit; d / ||d|| không đổi
    # khi cùng dịch mọi phần tử. eps của ONNX (1e-12) bị bỏ qua.
    peak = np.abs(deviation).max(axis=-1, keepdims=True)
    shift_down = np.maximum(bit_length(peak) + 1 - SQUARE_BITS, 0)
    reduced = check_bits(deviation >> shift_down, SQUARE_BITS, "deviation")
    square_sum = (reduced * reduced).sum(axis=-1, keepdims=True)
    check_bits(square_sum, ACCUMULATOR_BITS, "tổng bình phương")

    # Vector hằng có d = 0, khi đó đầu ra chỉ còn beta.
    constant = square_sum == 0
    mantissa, shift = rsqrt_fixed(np.where(constant, 1, square_sum), lut)

    # d / ||d||, bão hoà vì có thể chạm đúng 1.0 khi chỉ một phần tử khác 0.
    norm_limit = (1 << NORM_FRAC_BITS) - 1
    norm = round_shift(reduced * mantissa, shift - NORM_FRAC_BITS)
    norm = np.where(constant, 0, np.clip(norm, -norm_limit, norm_limit))

    scaled = round_shift(
        norm * gamma_fixed,
        NORM_FRAC_BITS + GAMMA_FRAC_BITS - LAYERNORM_OUT_FRAC_BITS,
    )

    return check_bits(scaled + beta_fixed, 32, "đầu ra LayerNorm").astype(np.int32)
