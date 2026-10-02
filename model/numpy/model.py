import numpy as np

from .config import (
    HEAD_DIM,
    HIDDEN_SIZE,
    LAYER_NORM_EPS,
    MAX_POSITION,
    NUM_HEADS,
    NUM_LAYERS,
)
from .ops import gelu, layer_norm, quantized_linear, softmax
from .weights import load_weights


class QuantizedLinear:
    def __init__(self, weights, prefix):
        self.weight = weights[f"{prefix}.weight"]
        self.weight_scale = weights[f"{prefix}.weight_scale"]
        self.weight_zero_point = weights[f"{prefix}.weight_zero_point"]
        self.bias = weights[f"{prefix}.bias"]

    def __call__(self, x, trace=None, name="Quantized linear"):
        return quantized_linear(
            x,
            self.weight,
            self.weight_scale,
            self.weight_zero_point,
            self.bias,
            trace,
            name,
        )


class Embeddings:
    def __init__(self, weights):
        self.word = weights["embeddings.word_embeddings.weight"]
        self.position = weights["embeddings.position_embeddings.weight"]
        self.token_type = weights["embeddings.token_type_embeddings.weight"]
        self.layer_norm_weight = weights["embeddings.layer_norm.weight"]
        self.layer_norm_bias = weights["embeddings.layer_norm.bias"]

    def __call__(self, input_ids, token_type_ids, trace=None):
        sequence_length = input_ids.shape[1]
        position_ids = np.arange(sequence_length, dtype=np.int64)[None, :]
        word = self.word[input_ids]
        token_type = self.token_type[token_type_ids]
        word_plus_type = word + token_type
        position = self.position[position_ids]
        hidden_state = word_plus_type + position
        if trace:
            trace.section("Embeddings", 2)
            trace.operation(
                "Embedding lookup and addition",
                "hidden=word[input_ids]+token_type[token_type_ids]+position[position_ids]",
            )
            trace.tensor("input_ids", input_ids)
            trace.tensor("token_type_ids", token_type_ids)
            trace.tensor("position_ids", position_ids)
            trace.tensor("word_lookup", word)
            trace.tensor("token_type_lookup", token_type)
            trace.tensor("word_plus_token_type", word_plus_type)
            trace.tensor("position_lookup", position)
            trace.tensor("embedding_sum", hidden_state)
        return layer_norm(
            hidden_state,
            self.layer_norm_weight,
            self.layer_norm_bias,
            LAYER_NORM_EPS,
            trace,
            "Embedding LayerNorm",
        )


class SelfAttention:
    def __init__(self, weights, prefix):
        self.query = QuantizedLinear(weights, f"{prefix}.query")
        self.key = QuantizedLinear(weights, f"{prefix}.key")
        self.value = QuantizedLinear(weights, f"{prefix}.value")

    @staticmethod
    def split_heads(x):
        batch_size, sequence_length, _ = x.shape
        return x.reshape(
            batch_size,
            sequence_length,
            NUM_HEADS,
            HEAD_DIM,
        ).transpose(0, 2, 1, 3)

    def __call__(self, x, attention_mask, trace=None):
        if trace:
            trace.section("Self-Attention", 3)
        query_linear = self.query(x, trace, "Query projection")
        key_linear = self.key(x, trace, "Key projection")
        value_linear = self.value(x, trace, "Value projection")
        query = self.split_heads(query_linear)
        key = self.split_heads(key_linear)
        value = self.split_heads(value_linear)

        if trace:
            trace.operation(
                "Split attention heads",
                "[batch,sequence,hidden] -> [batch,heads,sequence,head_dim]",
            )
            trace.tensor("query_heads", query)
            trace.tensor("key_heads", key)
            trace.tensor("value_heads", value)

        attention_scale = np.sqrt(
            np.float32(1.0) / np.sqrt(np.float32(HEAD_DIM)),
            dtype=np.float32,
        )
        scaled_query = query * attention_scale
        scaled_key = key * attention_scale
        transposed_key = scaled_key.swapaxes(-2, -1)
        scores = np.matmul(scaled_query, transposed_key)
        mask = np.where(
            attention_mask[:, None, None, :] == 0,
            np.float32(np.finfo(np.float32).min),
            np.float32(0.0),
        )
        masked_scores = scores + mask
        if trace:
            trace.operation(
                "Scaled query-key product and mask",
                "scores=(query*scale)@(key*scale)^T; masked_scores=scores+mask",
            )
            trace.scalar("attention_scale", attention_scale)
            trace.tensor("scaled_query", scaled_query)
            trace.tensor("scaled_key", scaled_key)
            trace.tensor("transposed_key", transposed_key)
            trace.tensor("scores", scores)
            trace.tensor("attention_mask", attention_mask)
            trace.tensor("additive_mask", mask)
            trace.tensor("masked_scores", masked_scores)
        probabilities = softmax(
            masked_scores, trace=trace, name="Attention probabilities"
        )
        context_heads = np.matmul(probabilities, value)
        context = context_heads.transpose(0, 2, 1, 3)
        output = context.reshape(x.shape[0], x.shape[1], HIDDEN_SIZE)
        if trace:
            trace.operation(
                "Attention context",
                "context=reshape(transpose(probabilities@value))",
            )
            trace.tensor("context_heads", context_heads)
            trace.tensor("transposed_context", context)
            trace.tensor("output", output)
        return output


class AttentionOutput:
    def __init__(self, weights, prefix):
        self.dense = QuantizedLinear(weights, f"{prefix}.dense")
        self.layer_norm_weight = weights[f"{prefix}.layer_norm.weight"]
        self.layer_norm_bias = weights[f"{prefix}.layer_norm.bias"]

    def __call__(self, attention, residual, trace=None):
        if trace:
            trace.section("Attention Output", 3)
        projected = self.dense(attention, trace, "Attention output projection")
        residual_sum = projected + residual
        if trace:
            trace.operation("Attention residual addition", "sum=projection+residual")
            trace.tensor("projection", projected)
            trace.tensor("residual", residual)
            trace.tensor("sum", residual_sum)
        return layer_norm(
            residual_sum,
            self.layer_norm_weight,
            self.layer_norm_bias,
            LAYER_NORM_EPS,
            trace,
            "Attention LayerNorm",
        )


class FeedForward:
    def __init__(self, weights, prefix):
        self.dense1 = QuantizedLinear(weights, f"{prefix}.dense1")
        self.dense2 = QuantizedLinear(weights, f"{prefix}.dense2")
        self.layer_norm_weight = weights[f"{prefix}.layer_norm.weight"]
        self.layer_norm_bias = weights[f"{prefix}.layer_norm.bias"]

    def __call__(self, x, trace=None):
        if trace:
            trace.section("Feed Forward", 3)
        residual = x
        intermediate = self.dense1(x, trace, "Feed-forward input projection")
        activated = gelu(intermediate, trace, "Feed-forward GELU")
        x = self.dense2(activated, trace, "Feed-forward output projection")
        residual_sum = x + residual
        if trace:
            trace.operation("Feed-forward residual addition", "sum=output+residual")
            trace.tensor("output", x)
            trace.tensor("residual", residual)
            trace.tensor("sum", residual_sum)
        return layer_norm(
            residual_sum,
            self.layer_norm_weight,
            self.layer_norm_bias,
            LAYER_NORM_EPS,
            trace,
            "Feed-forward LayerNorm",
        )


class EncoderLayer:
    def __init__(self, weights, layer):
        prefix = f"layers.{layer}"
        self.self_attention = SelfAttention(weights, f"{prefix}.self_attention")
        self.attention_output = AttentionOutput(
            weights,
            f"{prefix}.attention_output",
        )
        self.feed_forward = FeedForward(weights, f"{prefix}.ffn")

    def __call__(self, x, attention_mask, trace=None):
        attention = self.self_attention(x, attention_mask, trace)
        x = self.attention_output(attention, x, trace)
        return self.feed_forward(x, trace)


class MiniLM:
    def __init__(self, weights):
        self.embeddings = Embeddings(weights)
        self.layers = [
            EncoderLayer(weights, layer)
            for layer in range(NUM_LAYERS)
        ]

    @classmethod
    def from_onnx(cls, path):
        return cls(load_weights(path))

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, trace=None):
        input_ids = np.asarray(input_ids, dtype=np.int64)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > MAX_POSITION:
            raise ValueError(f"sequence length exceeds {MAX_POSITION} positions")

        if attention_mask is None:
            attention_mask = np.ones_like(input_ids)
        else:
            attention_mask = np.asarray(attention_mask, dtype=np.int64)
        if token_type_ids is None:
            token_type_ids = np.zeros_like(input_ids)
        else:
            token_type_ids = np.asarray(token_type_ids, dtype=np.int64)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape")
        if token_type_ids.shape != input_ids.shape:
            raise ValueError("token_type_ids must match input_ids shape")

        if trace:
            trace.section("Input", 2)
            trace.tensor("input_ids", input_ids)
            trace.tensor("attention_mask", attention_mask)
            trace.tensor("token_type_ids", token_type_ids)

        hidden_state = self.embeddings(input_ids, token_type_ids, trace)
        for index, layer in enumerate(self.layers):
            if trace:
                trace.section(f"Encoder Layer {index}", 2)
            hidden_state = layer(hidden_state, attention_mask, trace)

        if trace:
            trace.section("Final Output", 2)
            trace.tensor("last_hidden_state", hidden_state)
        return hidden_state.astype(np.float32, copy=False)
