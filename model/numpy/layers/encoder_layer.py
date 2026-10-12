if __package__:
    from .attention_output import AttentionOutput
    from .self_attention import SelfAttention
    from .feed_forward import FeedForward
else:
    from attention_output import AttentionOutput
    from self_attention import SelfAttention
    from feed_forward import FeedForward

class EncoderLayer:
    def __init__(self, tensors):
        self.tensors = tensors
        self.self_attention = SelfAttention(tensors)
        self.attention_output = AttentionOutput(tensors)
        self.feed_forward = FeedForward(tensors)

    def __call__(self, x, attention_mask=None):

        raise NotImplementedError
