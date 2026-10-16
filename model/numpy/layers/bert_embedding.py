import numpy as np

from .layer_norm import LayerNorm

if __package__ == "layers":
    from utils.requant import symmetric_quantize
else:
    from ..utils.requant import symmetric_quantize


class BertEmbeddings:
    """MiniLM embeddings and LayerNorm, followed by symmetric int8 quantization."""

    def __init__(self, tensors, trace=False, layers_output=None):
        self.tensors = tensors
        self.trace = trace
        self.layers_output = layers_output

        self.layer_norm = LayerNorm(tensors)

    def _lookup(self, name, ids):
        prefix = f"embeddings.{name}.weight"
        quantized = self.tensors[f"{prefix}_quantized"][ids]
        zero_point = np.float32(self.tensors[f"{prefix}_zero_point"])
        scale = np.float32(self.tensors[f"{prefix}_scale"])
        return (quantized.astype(np.float32) - zero_point) * scale

    def forward(self, input_ids, token_type_ids=None):
        _, seq_len = input_ids.shape
        if token_type_ids is None:
            token_type_ids = np.zeros_like(input_ids)
        position_ids = np.arange(seq_len, dtype=np.int64)[None, :]

        word = self._lookup("word_embeddings", input_ids)
        position = self._lookup("position_embeddings", position_ids)
        token_type = self._lookup("token_type_embeddings", token_type_ids)

        summed = word + position + token_type
        output = self.layer_norm(summed)
        quantized, scale_exponent = symmetric_quantize(output)

        # TRACE
        if self.trace and self.layers_output is not None:
            self.layers_output.update({
                "embeddings.word": word,
                "embeddings.position": position,
                "embeddings.token_type": token_type,
                "embeddings.sum": summed,
                "embeddings.output": output,
                "embeddings.quantized": quantized,
                "embeddings.scale_exponent": scale_exponent,
            })
        return quantized, scale_exponent

    def __call__(self, input_ids, token_type_ids=None):
        return self.forward(input_ids, token_type_ids)
