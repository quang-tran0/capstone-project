from ..config import NUM_LAYERS
from .int_layer_norm import IntLayerNorm
from .linear import Linear


class AttentionOutput:
    """Attention projection, Q.16 residual addition, then LayerNorm."""

    def __init__(self, tensors, layer_index=0, trace=False, layers_output=None):
        if not 0 <= layer_index < NUM_LAYERS:
            raise ValueError(f"layer_index must be between 0 and {NUM_LAYERS - 1}")
        prefix = f"encoder.layer.{layer_index}.attention.output"
        self.dense = Linear(
            tensors[f"{prefix}.dense.weight"], tensors[f"{prefix}.dense.bias"],
            name=f"{layer_index}.attention.output.dense",
            trace=trace, layers_output=layers_output,
        )
        self.layer_norm = IntLayerNorm(
            tensors, prefix=f"{prefix}.LayerNorm",
            trace=trace, layers_output=layers_output,
        )

    def __call__(self, attention_output, hidden_states):
        projected = self.dense(attention_output)
        # Both operands are Q.16, so the residual is a plain integer add.
        return self.layer_norm(projected + hidden_states)
