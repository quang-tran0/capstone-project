import numpy as np

from ..utils.quantize import symmetric_dequantize, symmetric_quantize


class IntLayerNorm:
    """Integer/ScaleInt12 LayerNorm with exact integer centering.

    Integer statistics precede float64 reciprocal sqrt and affine parameters.
    This is a NumPy reference, not an integer-only sqrt approximation.
    """

    def __init__(self, tensors, prefix="embeddings.LayerNorm", eps=1e-12,
                 trace=False, layers_output=None):
        self.weight = np.asarray(tensors[f"{prefix}.weight"], dtype=np.float64)
        self.bias = np.asarray(tensors[f"{prefix}.bias"], dtype=np.float64)
        self.eps = eps
        self.prefix = prefix
        self.trace = trace
        self.layers_output = layers_output

    def __call__(self, hidden_states):
        codes, scale = hidden_states
        codes = np.asarray(codes)
        if not np.issubdtype(codes.dtype, np.integer):
            raise TypeError("LayerNorm input must contain integer codes")
        if codes.shape[-1] != self.weight.size or float(scale) <= 0:
            raise ValueError("Invalid LayerNorm hidden dimension or scale")
        width = codes.shape[-1]
        codes = codes.astype(np.int64)
        centered = codes * width - codes.sum(axis=-1, keepdims=True)
        variance = np.mean(centered.astype(np.float64)**2, axis=-1, keepdims=True)
        denominator = np.sqrt(variance + self.eps * (width / float(scale))**2)
        output = centered / denominator * self.weight + self.bias
        result = symmetric_quantize(output)
        if self.trace and self.layers_output is not None:
            self.layers_output.update({
                f"{self.prefix}.output": symmetric_dequantize(*result),
                f"{self.prefix}.quantized": result[0],
                f"{self.prefix}.scale": result[1],
            })
        return result
