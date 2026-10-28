import numpy as np

from ..utils.fixed import FRAC_BITS, check_bits, round_shift, to_float
from ..utils.int_quantize import quantize_fixed, quantize_weight


# INT8 x INT8 MACs plus nothing else: K <= 1536 products of |x| <= 127^2.
ACCUMULATOR_BITS = 32


class Linear:
    """INT8 matmul with per-token input scales and per-channel weight scales.

    y[i, j] = acc[i, j] * S_x[i] * S_w[j] + b[j], returned in Q.16.
    """

    def __init__(self, weight, bias, name, trace=False, layers_output=None):
        # Offline: weight from ONNX has shape (in_features, out_features).
        self.weight_q, self.weight_m, self.weight_n = quantize_weight(weight, axis=0)
        self.bias = np.round(np.asarray(bias, dtype=np.float64) * (1 << FRAC_BITS)).astype(np.int64)

        self.name = name
        self.trace = trace
        self.layers_output = layers_output

    def forward_quantized(self, x_q, x_m, x_n):
        """x_q: INT8 (..., in) with scale fields (m, n) of shape (..., 1)."""
        accumulator = np.matmul(x_q.astype(np.int64), self.weight_q.astype(np.int64))
        check_bits(accumulator, ACCUMULATOR_BITS, f"{self.name} accumulator")

        # S_x * S_w = (m_x * m_w) * 2**-(n_x + n_w), rescaled to Q.16.
        output = round_shift(
            accumulator * (x_m * self.weight_m),
            x_n + self.weight_n - FRAC_BITS,
        ) + self.bias
        output = check_bits(output, 32, f"{self.name} output").astype(np.int32)

        if self.trace and self.layers_output is not None:
            self.layers_output.update({
                f"encoder.layer.{self.name}.accumulator": accumulator,
                f"encoder.layer.{self.name}.output": to_float(output),
            })
        return output

    def __call__(self, x):
        """x: Q.16 (..., in) -> Q.16 (..., out); the input is quantized per token."""
        x = np.asarray(x)
        if not np.issubdtype(x.dtype, np.integer):
            raise TypeError("Linear input must be a Q.16 integer tensor")
        return self.forward_quantized(*quantize_fixed(x, axis=-1))
