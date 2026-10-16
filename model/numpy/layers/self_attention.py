import numpy as np

from ..config import HEAD_DIM, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS
from ..utils.quantize import symmetric_dequantize, symmetric_quantize
from ..utils.softmax import softmax
from .linear import Linear


class SelfAttention:
    """Quantized Q/K/V and score MACs, with float32 softmax and context mixing."""

    def __init__(self, tensors, layer_index=0, trace=False, layers_output=None):
        if not 0 <= layer_index < NUM_LAYERS:
            raise ValueError(f"layer_index must be between 0 and {NUM_LAYERS - 1}")
        self.tensors = tensors
        self.layer_index = layer_index
        self.trace = trace
        self.layers_output = layers_output
        self.prefix = f"encoder.layer.{layer_index}.attention.self"

        self.query = self._linear("query")
        self.key = self._linear("key")
        self.value = self._linear("value")

    def _linear(self, name):
        prefix = f"{self.prefix}.{name}"
        return Linear(
            self.tensors[f"{prefix}.weight"],
            self.tensors[f"{prefix}.bias"],
            name=f"{self.layer_index}.attention.self.{name}",
            trace=self.trace,
            layers_output=self.layers_output,
        )

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

    def __call__(self, hidden_states, attention_mask=None):
        """
        hidden_states: (quantized tensor, scale), as returned by BertEmbeddings.
            Tensor shape: (batch_size, sequence_length, 384), codes in [-128, 127].
        attention_mask: (batch_size, sequence_length)

        Returns: (quantized context, scale), with tensor shape (B, N, 384).
        Integer storage dtype follows symmetric_quantize (currently int32).
        """
        x_quantized, x_scale = hidden_states
        x_quantized = np.asarray(x_quantized)
        if x_quantized.ndim != 3 or x_quantized.shape[-1] != HIDDEN_SIZE:
            raise ValueError(f"Expected quantized input shape (B, N, {HIDDEN_SIZE})")
        if not np.issubdtype(x_quantized.dtype, np.integer):
            raise TypeError("hidden_states must contain integer quantized codes")
        if np.any(x_quantized < -128) or np.any(x_quantized > 127):
            raise ValueError("Input codes must be in [-128, 127]")
        if not np.isfinite(x_scale) or x_scale <= 0:
            raise ValueError("Input scale must be finite and positive")
        if attention_mask is not None:
            attention_mask = np.asarray(attention_mask)
            if attention_mask.shape != x_quantized.shape[:2]:
                raise ValueError("attention_mask must have shape (batch_size, sequence_length)")

        x_quantized = x_quantized.astype(np.int32, copy=False)
        q, q_scale = self.query(x_quantized, x_scale)
        k, k_scale = self.key(x_quantized, x_scale)
        v, v_scale = self.value(x_quantized, x_scale)
        q = self.split_heads(q).astype(np.int32, copy=False)
        k = self.split_heads(k).astype(np.int32, copy=False)
        v = self.split_heads(v)

        score_accumulator = q @ k.swapaxes(-2, -1)
        scores = score_accumulator.astype(np.float32) * np.float32(q_scale * k_scale)
        scores = scores / np.float32(np.sqrt(HEAD_DIM))
        if attention_mask is not None:
            scores = np.where(
                attention_mask[:, None, None, :] == 0,
                np.finfo(np.float32).min,
                scores,
            )
        probabilities = softmax(scores, axis=-1)
        context = probabilities @ symmetric_dequantize(v, v_scale)
        batch_size, seq_len, _ = x_quantized.shape
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, HIDDEN_SIZE)
        quantized, scale = symmetric_quantize(context)

        if self.trace and self.layers_output is not None:
            self.layers_output.update({
                f"{self.prefix}.query_heads": symmetric_dequantize(q, q_scale),
                f"{self.prefix}.key_heads": symmetric_dequantize(k, k_scale),
                f"{self.prefix}.value_heads": symmetric_dequantize(v, v_scale),
                f"{self.prefix}.score_accumulator": score_accumulator,
                f"{self.prefix}.scores": scores,
                f"{self.prefix}.probabilities": probabilities,
                f"{self.prefix}.context": context,
                f"{self.prefix}.output": symmetric_dequantize(quantized, scale),
                f"{self.prefix}.quantized": quantized,
                f"{self.prefix}.scale": scale,
            })
        return quantized, scale
