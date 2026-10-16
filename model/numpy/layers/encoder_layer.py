if __package__:
    from .attention_output import AttentionOutput
    from .self_attention import SelfAttention
    from .feed_forward import FeedForward
else:
    from attention_output import AttentionOutput
    from self_attention import SelfAttention
    from feed_forward import FeedForward

class EncoderLayer:
    def __init__(self, tensors, trace=False, layers_output=None):
        self.tensors = tensors
        self.trace = trace
        self.layers_output = layers_output
        self.self_attention = SelfAttention(tensors, trace=trace, layers_output=layers_output)
        self.attention_output = AttentionOutput(tensors, trace=trace, layers_output=layers_output)
        self.feed_forward = FeedForward(tensors, trace=trace, layers_output=layers_output)

    def __call__(self, x, attention_mask=None):
        attention = self.self_attention(
            x,
            attention_mask,
        )

        x = self.attention_output(
            attention,
            x,
        )

        x = self.feed_forward(x)

        return x
