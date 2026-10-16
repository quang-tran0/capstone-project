import numpy as np

from .layer_norm import LayerNorm
from ..utils.quantize import symmetric_quantize, symmetric_dequantize

class BertEmbeddings:
    """Float32 embeddings and LayerNorm, followed by symmetric INT8 quantization."""

    def __init__(self, tensors, trace=False, layers_output=None):
        self.tensors = tensors
        self.trace = trace
        self.layers_output = layers_output

        self.layer_norm = LayerNorm(tensors)

    def _lookup(self, name, ids):
        return self.tensors[f"embeddings.{name}.weight"][ids]

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

        quantized_output, scale = symmetric_quantize(output)

        # TRACE
        if self.trace and self.layers_output is not None:
            self.layers_output.update({
                "embeddings.word": word,
                "embeddings.position": position,
                "embeddings.token_type": token_type,
                "embeddings.sum": summed,
                "embeddings.output": symmetric_dequantize(quantized_output, scale),
                "embeddings.quantized": quantized_output,
                "embeddings.scale": scale,
            })
        return quantized_output, scale

    def display_trace(self):
        if self.trace and self.layers_output is not None:
            print("\n===== Bert Embeddings Layer Info =====")
            print("- Output max:", np.max(self.layers_output["embeddings.output"]))
            print("- Output min:", np.min(self.layers_output["embeddings.output"]))
            print("- Output mean:", np.mean(self.layers_output["embeddings.output"]))
            print("- Scale:", self.layers_output["embeddings.scale"])

    def __call__(self, input_ids, token_type_ids=None):
        return self.forward(input_ids, token_type_ids)
