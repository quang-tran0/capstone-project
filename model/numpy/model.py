import numpy as np

from .config import NUM_LAYERS
from .weights import load_initializers
from .layers.bert_embedding import BertEmbeddings
from .layers.encoder_layer import EncoderLayer
from .layers.mean_pooling import MeanPooling


class SentenceEmbeddingModel:
    """MiniLM embeddings, all encoder blocks and masked mean pooling."""

    def __init__(self, model_path, trace=False):
        self.model_path = model_path
        self.trace = trace
        self.tensors = load_initializers(model_path)
        self.layers_output = {}
        self.bert_embeddings = BertEmbeddings(
            self.tensors, trace=trace, layers_output=self.layers_output,
        )
        self.layers = [
            EncoderLayer(self.tensors, index, trace=trace, layers_output=self.layers_output)
            for index in range(NUM_LAYERS)
        ]
        self.pooling = MeanPooling()

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None):
        """Return Q.16 sentence embeddings of shape (B, 384) as raw np.int32."""
        input_ids = np.asarray(input_ids)
        if input_ids.ndim != 2 or 0 in input_ids.shape:
            raise ValueError("input_ids must have non-empty shape (B, N)")
        if not np.issubdtype(input_ids.dtype, np.integer):
            raise TypeError("input_ids must be integer token IDs")
        if attention_mask is not None:
            attention_mask = np.asarray(attention_mask)
            if attention_mask.shape != input_ids.shape or not np.all(
                (attention_mask == 0) | (attention_mask == 1)
            ):
                raise ValueError("attention_mask must be binary with shape (B, N)")
        self.layers_output.clear()
        hidden_states = self.bert_embeddings(input_ids, token_type_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        result = self.pooling(hidden_states, attention_mask)
        if self.trace:
            self.layers_output.update({
                "encoder.output": hidden_states.to_float(),
                "pooling.output": result.to_float(),
            })
        return result.raw.astype(np.int32)

    def display_trace(self):
        for name, value in self.layers_output.items():
            if isinstance(value, np.ndarray) and name.endswith(".output"):
                print(f"{name}: shape={value.shape}, min={value.min():.6g}, max={value.max():.6g}")
