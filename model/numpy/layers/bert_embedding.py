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
        input_ids = np.asarray(input_ids)
        if input_ids.ndim != 2 or 0 in input_ids.shape:
            raise ValueError("input_ids must have non-empty shape (B, N)")
        if not np.issubdtype(input_ids.dtype, np.integer):
            raise TypeError("input_ids must contain integer token IDs")
        vocabulary = self.tensors["embeddings.word_embeddings.weight"].shape[0]
        if np.any(input_ids < 0) or np.any(input_ids >= vocabulary):
            raise ValueError("input_ids contain an out-of-vocabulary token ID")
        _, seq_len = input_ids.shape
        if seq_len > self.tensors["embeddings.position_embeddings.weight"].shape[0]:
            raise ValueError("Sequence exceeds the position embedding table")
        if token_type_ids is None:
            token_type_ids = np.zeros_like(input_ids)
        token_type_ids = np.asarray(token_type_ids)
        if token_type_ids.shape != input_ids.shape or not np.issubdtype(token_type_ids.dtype, np.integer):
            raise ValueError("token_type_ids must have integer shape (B, N)")
        type_count = self.tensors["embeddings.token_type_embeddings.weight"].shape[0]
        if np.any(token_type_ids < 0) or np.any(token_type_ids >= type_count):
            raise ValueError("token_type_ids contain an out-of-range type ID")
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
