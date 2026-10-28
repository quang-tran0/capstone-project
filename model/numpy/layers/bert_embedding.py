import numpy as np

from .int_layer_norm import IntLayerNorm
from ..utils.fixed import FRAC_BITS, round_shift, to_float
from ..utils.int_quantize import quantize_weight

EMBEDDING_NAMES = ("word_embeddings", "position_embeddings", "token_type_embeddings")


class BertEmbeddings:
    """INT8 embedding tables (one scale per row), Q.16 sum and integer LayerNorm."""

    def __init__(self, tensors, trace=False, layers_output=None):
        # Offline: each table row gets its own INT8 scale m * 2**-n.
        self.tables = {
            name: quantize_weight(tensors[f"embeddings.{name}.weight"], axis=1)
            for name in EMBEDDING_NAMES
        }
        self.trace = trace
        self.layers_output = layers_output

        self.layer_norm = IntLayerNorm(tensors)

    def _lookup(self, name, ids):
        """Read INT8 rows and rescale them to Q.16: q * m * 2**(16 - n)."""
        codes, multiplier, degree = self.tables[name]
        return round_shift(
            codes[ids].astype(np.int64) * multiplier[ids],
            degree[ids] - FRAC_BITS,
        )

    def forward(self, input_ids, token_type_ids=None):
        input_ids = np.asarray(input_ids)
        if input_ids.ndim != 2 or 0 in input_ids.shape:
            raise ValueError("input_ids must have non-empty shape (B, N)")
        if not np.issubdtype(input_ids.dtype, np.integer):
            raise TypeError("input_ids must contain integer token IDs")
        vocabulary = self.tables["word_embeddings"][0].shape[0]
        if np.any(input_ids < 0) or np.any(input_ids >= vocabulary):
            raise ValueError("input_ids contain an out-of-vocabulary token ID")
        _, seq_len = input_ids.shape
        if seq_len > self.tables["position_embeddings"][0].shape[0]:
            raise ValueError("Sequence exceeds the position embedding table")
        if token_type_ids is None:
            token_type_ids = np.zeros_like(input_ids)
        token_type_ids = np.asarray(token_type_ids)
        if token_type_ids.shape != input_ids.shape or not np.issubdtype(token_type_ids.dtype, np.integer):
            raise ValueError("token_type_ids must have integer shape (B, N)")
        type_count = self.tables["token_type_embeddings"][0].shape[0]
        if np.any(token_type_ids < 0) or np.any(token_type_ids >= type_count):
            raise ValueError("token_type_ids contain an out-of-range type ID")
        position_ids = np.arange(seq_len, dtype=np.int64)[None, :]

        word = self._lookup("word_embeddings", input_ids)
        position = self._lookup("position_embeddings", position_ids)
        token_type = self._lookup("token_type_embeddings", token_type_ids)

        summed = word + position + token_type
        output = self.layer_norm(summed)

        # TRACE
        if self.trace and self.layers_output is not None:
            self.layers_output.update({
                "embeddings.word": to_float(word),
                "embeddings.position": to_float(np.broadcast_to(position, word.shape)),
                "embeddings.token_type": to_float(token_type),
                "embeddings.sum": to_float(summed),
                "embeddings.output": to_float(output),
            })
        return output

    def display_trace(self):
        if self.trace and self.layers_output is not None:
            print("\n===== Bert Embeddings Layer Info =====")
            print("- Output max:", np.max(self.layers_output["embeddings.output"]))
            print("- Output min:", np.min(self.layers_output["embeddings.output"]))
            print("- Output mean:", np.mean(self.layers_output["embeddings.output"]))

    def __call__(self, input_ids, token_type_ids=None):
        return self.forward(input_ids, token_type_ids)
