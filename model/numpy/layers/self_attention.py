import math

import numpy as np

from ..config import HEAD_DIM, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS
from ..hwtypes import Fixed, Scale
from ..utils.fixed import FRAC_BITS, WIDE_BITS
from ..utils.int_quantize import quantize
from ..utils.reciprocal import prepare_reciprocal_lut, reciprocal_fixed
from ..utils.softmax import PROBABILITY_MAX, prepare_exp2_lut, softmax_fixed
from .linear import Linear


# 1/sqrt(HEAD_DIM) as a constant scale m * 2**-15, folded into the score rescale.
INV_SQRT_HEAD_DIM_BITS = 15
INV_SQRT_HEAD_DIM = Scale.const(
    round((1 << INV_SQRT_HEAD_DIM_BITS) / math.sqrt(HEAD_DIM)), INV_SQRT_HEAD_DIM_BITS,
)
SCORE_ACCUMULATOR_BITS = 20
CONTEXT_ACCUMULATOR_BITS = 22


class SelfAttention:
    """Integer Q/K/V, scores, softmax and context; output in Q.16."""

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
        self.exp2_lut = prepare_exp2_lut()
        self.reciprocal_lut = prepare_reciprocal_lut()

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
        hidden_states: Fixed Q.16 (batch_size, sequence_length, 384).
        attention_mask: (batch_size, sequence_length), 1 for real tokens.

        Returns: Q.16 context of shape (B, N, 384).
        """
        if not isinstance(hidden_states, Fixed):
            raise TypeError("hidden_states must be a Fixed Q.16 tensor")
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != HIDDEN_SIZE:
            raise ValueError(f"Expected input shape (B, N, {HIDDEN_SIZE})")
        batch_size, seq_len, _ = hidden_states.shape
        if attention_mask is None:
            attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
        attention_mask = np.asarray(attention_mask)
        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError("attention_mask must have shape (batch_size, sequence_length)")
        if not np.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must contain only 0 or 1")

        # The input is quantized once and shared by Q, K and V.
        x = quantize(hidden_states, axis=-1)
        q = self.split_heads(self.query.forward_quantized(x))
        k = self.split_heads(self.key.forward_quantized(x))
        v = self.split_heads(self.value.forward_quantized(x))

        # Q and K: one scale per token and head. V: one scale per channel,
        # since the context sums over tokens.
        q = quantize(q, axis=-1)
        k = quantize(k, axis=-1)
        v = quantize(v, axis=-2)

        score_accumulator = (q.codes @ k.codes.swapaxes(-2, -1)).resize(
            SCORE_ACCUMULATOR_BITS, name="score accumulator")

        # z = acc * S_q[i] * S_k[j] / sqrt(D), in Q.16.
        score_scale = q.scale * k.scale.swapaxes(-2, -1) * INV_SQRT_HEAD_DIM
        scores = score_scale.rescale(score_accumulator, FRAC_BITS)

        mask = attention_mask[:, None, None, :] != 0
        probabilities, total = softmax_fixed(scores, mask, self.exp2_lut)

        context_accumulator = (probabilities @ v.codes).resize(
            CONTEXT_ACCUMULATOR_BITS, name="context accumulator")

        # context = acc * S_v / (127 * E), one reciprocal per row; E is Q.15.
        inverse_total = reciprocal_fixed(PROBABILITY_MAX * total, self.reciprocal_lut)
        context = (v.scale * inverse_total).rescale(context_accumulator, FRAC_BITS)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, HIDDEN_SIZE)
        context = context.resize(WIDE_BITS, name="context")

        if self.trace and self.layers_output is not None:
            self.layers_output.update({
                f"{self.prefix}.score_accumulator": score_accumulator.raw,
                f"{self.prefix}.scores": scores.to_float(),
                f"{self.prefix}.probabilities_quantized": probabilities.raw,
                f"{self.prefix}.probability_total": total.raw,
                f"{self.prefix}.context_accumulator": context_accumulator.raw,
                f"{self.prefix}.output": context.to_float(),
            })
        return context
