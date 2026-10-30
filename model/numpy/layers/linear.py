from ..hwtypes import Fixed, QTensor
from ..utils.fixed import FRAC_BITS, WIDE_BITS
from ..utils.int_quantize import quantize, quantize_weight


# INT8 x INT8 MACs plus nothing else: K <= 1536 products of |x| <= 127^2.
ACCUMULATOR_BITS = 32


class Linear:
    """INT8 matmul with per-token input scales and per-channel weight scales.

    y[i, j] = acc[i, j] * S_x[i] * S_w[j] + b[j], returned in Q.16.
    """

    def __init__(self, weight, bias, name, trace=False, layers_output=None):
        # Offline: weight from ONNX has shape (in_features, out_features).
        self.weight = quantize_weight(weight, axis=0)
        self.bias = Fixed.from_float(bias, WIDE_BITS, FRAC_BITS, name=f"{name} bias")

        self.name = name
        self.trace = trace
        self.layers_output = layers_output

    def forward_quantized(self, x: QTensor):
        """x: INT8 codes (..., in) with one scale per token, shape (..., 1)."""
        accumulator = (x.codes @ self.weight.codes).resize(
            ACCUMULATOR_BITS, name=f"{self.name} accumulator")

        # S_x * S_w = (m_x * m_w) * 2**-(n_x + n_w), rescaled to Q.16.
        output = (x.scale * self.weight.scale).rescale(accumulator, FRAC_BITS) + self.bias
        output = output.resize(WIDE_BITS, name=f"{self.name} output")

        if self.trace and self.layers_output is not None:
            self.layers_output.update({
                f"encoder.layer.{self.name}.accumulator": accumulator.raw,
                f"encoder.layer.{self.name}.output": output.to_float(),
            })
        return output

    def __call__(self, x: Fixed):
        """x: Q.16 (..., in) -> Q.16 (..., out); the input is quantized per token."""
        if not isinstance(x, Fixed):
            raise TypeError("Linear input must be a Fixed Q.16 tensor")
        return self.forward_quantized(quantize(x, axis=-1))
