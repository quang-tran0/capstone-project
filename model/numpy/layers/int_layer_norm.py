import math

import numpy as np

from ..utils.fixed import bit_length, check_bits, round_shift
from ..utils.quantize import requantize, symmetric_dequantize
from ..utils.rsqrt import prepare_rsqrt_lut, rsqrt_fixed
from ..utils.scale_int12 import ScaleInt12


# e = d >> shift fits the 18-bit DSP48E2 port before squaring.
SQUARE_BITS = 18
# Sum of squares, the width of the DSP48E2 accumulator.
ACCUMULATOR_BITS = 48
# d / ||d|| lies in [-1, 1].
NORM_FRAC_BITS = 17
# max |gamma| * sqrt(384) over all 25 LayerNorms is about 67, so Q.10 fits 18 bits.
GAMMA_FRAC_BITS = 10
OUTPUT_FRAC_BITS = 16


class IntLayerNorm:
    """Integer-only LayerNorm with exact integer centering.

    With d = N*x - sum(x), (x - mean) / std = sqrt(N) * d / ||d||, so there
    is no division; sqrt(N) is folded into gamma offline and 1/sqrt comes
    from a LUT plus one Newton step. The input scale cancels in d / ||d||,
    and eps (1e-12) is ignored. The Q.16 result is requantized to INT8.
    """

    def __init__(self, tensors, prefix="embeddings.LayerNorm",
                 trace=False, layers_output=None):
        weight = np.asarray(tensors[f"{prefix}.weight"], dtype=np.float64)
        bias = np.asarray(tensors[f"{prefix}.bias"], dtype=np.float64)
        self.width = weight.size

        # Offline constants: G = gamma * sqrt(N) in Q.10, B = beta in Q.16.
        gamma = np.round(weight * math.sqrt(self.width) * (1 << GAMMA_FRAC_BITS))
        self.gamma = check_bits(gamma.astype(np.int64), SQUARE_BITS, "gamma")
        self.beta = np.round(bias * (1 << OUTPUT_FRAC_BITS)).astype(np.int64)
        self.lut = prepare_rsqrt_lut()

        self.prefix = prefix
        self.trace = trace
        self.layers_output = layers_output

    def normalize(self, codes):
        """Return the LayerNorm output as np.int32 in Q.OUTPUT_FRAC_BITS."""
        codes = np.asarray(codes, dtype=np.int64)

        # d = N * (x - mean), without dividing by N.
        deviation = codes * self.width - codes.sum(axis=-1, keepdims=True)

        # Per-vector shift so e uses all SQUARE_BITS; d / ||d|| is unchanged
        # when every element is shifted by the same amount.
        peak = np.abs(deviation).max(axis=-1, keepdims=True)
        shift_down = np.maximum(bit_length(peak) + 1 - SQUARE_BITS, 0)
        reduced = check_bits(deviation >> shift_down, SQUARE_BITS, "deviation")
        square_sum = (reduced * reduced).sum(axis=-1, keepdims=True)
        check_bits(square_sum, ACCUMULATOR_BITS, "sum of squares")

        # A constant vector has d = 0, leaving only beta.
        constant = square_sum == 0
        mantissa, shift = rsqrt_fixed(np.where(constant, 1, square_sum), self.lut)

        # d / ||d||, saturated since it reaches exactly 1.0 when one element is nonzero.
        norm_limit = (1 << NORM_FRAC_BITS) - 1
        norm = round_shift(reduced * mantissa, shift - NORM_FRAC_BITS)
        norm = np.where(constant, 0, np.clip(norm, -norm_limit, norm_limit))

        scaled = round_shift(
            norm * self.gamma,
            NORM_FRAC_BITS + GAMMA_FRAC_BITS - OUTPUT_FRAC_BITS,
        )
        return check_bits(scaled + self.beta, 32, "LayerNorm output").astype(np.int32)

    def __call__(self, hidden_states):
        codes, scale = hidden_states
        codes = np.asarray(codes)
        if not np.issubdtype(codes.dtype, np.integer):
            raise TypeError("LayerNorm input must contain integer codes")
        if codes.shape[-1] != self.width or float(scale) <= 0:
            raise ValueError("Invalid LayerNorm hidden dimension or scale")
        output = self.normalize(codes)
        unit = ScaleInt12.from_parts(1, OUTPUT_FRAC_BITS, normalize=False)
        result = requantize(output, unit)
        if self.trace and self.layers_output is not None:
            self.layers_output.update({
                f"{self.prefix}.output": symmetric_dequantize(*result),
                f"{self.prefix}.quantized": result[0],
                f"{self.prefix}.scale": result[1],
            })
        return result
