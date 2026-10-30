from ..config import NUM_LAYERS
from ..utils.fixed import WIDE_BITS
from ..utils.gelu import gelu_fixed, prepare_phi_lut
from .int_layer_norm import IntLayerNorm
from .linear import Linear


class FeedForward:
    """Dense -> GELU (Phi LUT) -> dense -> Q.16 residual -> LayerNorm."""

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
        self.phi_lut = prepare_phi_lut()

    def __call__(self, hidden_states):
        # GELU works on the Q.16 accumulator output, before any INT8 rounding.
        activation = gelu_fixed(self.dense1(hidden_states), self.phi_lut).resize(
            WIDE_BITS, name="GELU output")
        if self.trace and self.layers_output is not None:
            self.layers_output[f"{self.prefix}.intermediate.gelu.output"] = activation.to_float()
        projected = self.dense2(activation)
        return self.layer_norm(projected + hidden_states)
