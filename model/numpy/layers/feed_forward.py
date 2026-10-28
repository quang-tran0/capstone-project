import math

import numpy as np

from ..config import NUM_LAYERS
from ..utils.quantize import add_quantized, symmetric_dequantize, symmetric_quantize
from .int_layer_norm import IntLayerNorm
from .linear import Linear


_erf = np.vectorize(math.erf, otypes=[np.float64])


class FeedForward:
    """Dense -> exact GELU -> dense -> residual -> LayerNorm."""

    def __init__(self, tensors, layer_index=0, trace=False, layers_output=None):
        if not 0 <= layer_index < NUM_LAYERS:
            raise ValueError(f"layer_index must be between 0 and {NUM_LAYERS - 1}")
        self.prefix = f"encoder.layer.{layer_index}"
        self.trace = trace
        self.layers_output = layers_output
        self.dense1 = Linear(
            tensors[f"{self.prefix}.intermediate.dense.weight"],
            tensors[f"{self.prefix}.intermediate.dense.bias"],
            name=f"{layer_index}.intermediate.dense",
            trace=trace, layers_output=layers_output,
        )
        self.dense2 = Linear(
            tensors[f"{self.prefix}.output.dense.weight"],
            tensors[f"{self.prefix}.output.dense.bias"],
            name=f"{layer_index}.output.dense", trace=trace, layers_output=layers_output,
        )
        self.layer_norm = IntLayerNorm(
            tensors, prefix=f"{self.prefix}.output.LayerNorm",
            trace=trace, layers_output=layers_output,
        )

    def __call__(self, hidden_states):
        intermediate = symmetric_dequantize(*self.dense1(*hidden_states))
        activated = intermediate * 0.5 * (1.0 + _erf(intermediate / math.sqrt(2.0)))
        activation = symmetric_quantize(activated)
        if self.trace and self.layers_output is not None:
            self.layers_output.update({
                f"{self.prefix}.intermediate.gelu.output": symmetric_dequantize(*activation),
                f"{self.prefix}.intermediate.gelu.quantized": activation[0],
                f"{self.prefix}.intermediate.gelu.scale": activation[1],
            })
        projected = self.dense2(*activation)
        return self.layer_norm(add_quantized(projected, hidden_states, wide=True))
