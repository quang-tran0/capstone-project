from .attention_output import AttentionOutput
from .self_attention import SelfAttention
from .feed_forward import FeedForward


class EncoderLayer:
    def __init__(self, tensors, layer_index=0, trace=False, layers_output=None):
        self.layer_index = layer_index
        self.trace = trace
        self.layers_output = layers_output
        self.self_attention = SelfAttention(
            tensors, layer_index, trace=trace, layers_output=layers_output,
        )
        self.attention_output = AttentionOutput(
            tensors, layer_index, trace=trace, layers_output=layers_output,
        )
        self.feed_forward = FeedForward(
            tensors, layer_index, trace=trace, layers_output=layers_output,
        )

    def __call__(self, x, attention_mask=None):
        attention = self.self_attention(x, attention_mask)
        x = self.attention_output(attention, x)
        x = self.feed_forward(x)
        if self.trace and self.layers_output is not None:
            self.layers_output[f"encoder.layer.{self.layer_index}.output"] = x
        return x
