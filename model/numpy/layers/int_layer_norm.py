import math

import numpy as np

from ..hwtypes import Fixed, leading_one, where
from ..utils.fixed import FRAC_BITS, WIDE_BITS
from ..utils.rsqrt import prepare_rsqrt_lut, rsqrt_fixed


# e = d >> shift fits the 18-bit DSP48E2 port before squaring.
SQUARE_BITS = 18
# Sum of squares, the width of the DSP48E2 accumulator.
ACCUMULATOR_BITS = 48
# d / ||d|| lies in [-1, 1].
NORM_FRAC_BITS = 17
# max |gamma| * sqrt(384) over all 25 LayerNorms is about 67, so Q.10 fits 18 bits.
GAMMA_FRAC_BITS = 10
OUTPUT_FRAC_BITS = FRAC_BITS


class IntLayerNorm:
    """Integer-only LayerNorm with exact integer centering.

    With d = N*x - sum(x), (x - mean) / std = sqrt(N) * d / ||d||, so there
    is no division; sqrt(N) is folded into gamma offline and 1/sqrt comes
    from a LUT plus one Newton step. The input scale cancels in d / ||d||,
    and eps (1e-12) is ignored. The output is the Q.16 residual stream.
    """

    def __init__(self, tensors, prefix="embeddings.LayerNorm",
                 trace=False, layers_output=None):
        weight = np.asarray(tensors[f"{prefix}.weight"], dtype=np.float64)
        bias = np.asarray(tensors[f"{prefix}.bias"], dtype=np.float64)
        self.width = weight.size

        # Offline constants: G = gamma * sqrt(N) in Q.10, B = beta in Q.16.
        self.gamma = Fixed.from_float(weight * math.sqrt(self.width), SQUARE_BITS,
                                      GAMMA_FRAC_BITS, name="gamma")
        self.beta = Fixed.from_float(bias, WIDE_BITS, OUTPUT_FRAC_BITS, name="beta")
        self.lut = prepare_rsqrt_lut()

        self.prefix = prefix
        self.trace = trace
        self.layers_output = layers_output

    def normalize(self, codes):
        """Return the LayerNorm output as Q.OUTPUT_FRAC_BITS in WIDE_BITS."""
        # d = N * (x - mean), without dividing by N.
        deviation = codes * self.width - codes.sum(axis=-1, keepdims=True)

        # Per-vector shift so e uses all SQUARE_BITS; d / ||d|| is unchanged
        # when every element is shifted by the same amount, so the fraction is dropped.
        peak = abs(deviation).max(axis=-1, keepdims=True)
        shift_down = (leading_one(peak) + (1 - SQUARE_BITS)).clip(0)
        reduced = (deviation >> shift_down).with_frac(0).resize(SQUARE_BITS, name="deviation")
        square_sum = (reduced * reduced).sum(axis=-1, keepdims=True).resize(
            ACCUMULATOR_BITS, name="sum of squares")

        # A constant vector has d = 0, leaving only beta.
        constant = square_sum == 0
        inverse_norm = rsqrt_fixed(where(constant, 1, square_sum), self.lut)

        # d / ||d||, saturated since it reaches exactly 1.0 when one element is nonzero.
        norm_limit = (1 << NORM_FRAC_BITS) - 1
        norm = inverse_norm.rescale(reduced, NORM_FRAC_BITS).clip(-norm_limit, norm_limit)
        norm = where(constant, 0, norm)

        scaled = (norm * self.gamma).round_shift(NORM_FRAC_BITS + GAMMA_FRAC_BITS - OUTPUT_FRAC_BITS)
        return (scaled + self.beta).resize(WIDE_BITS, name="LayerNorm output")

    def __call__(self, hidden_states):
        """Fixed input in any fraction -> Q.16."""
        if not isinstance(hidden_states, Fixed):
            raise TypeError("LayerNorm input must be a Fixed tensor")
        if hidden_states.shape[-1] != self.width:
            raise ValueError("Invalid LayerNorm hidden dimension")
        output = self.normalize(hidden_states)
        if self.trace and self.layers_output is not None:
            self.layers_output[f"{self.prefix}.output"] = output.to_float()
        return output
