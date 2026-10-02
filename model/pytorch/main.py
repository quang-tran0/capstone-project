import math
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper
from transformers import AutoTokenizer


HIDDEN_SIZE = 384
NUM_HEADS = 12
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS # 384 / 12 = 32
INTERMEDIATE_SIZE = 1536
NUM_LAYERS = 12
VOCAB_SIZE = 250037
MAX_POSITION = 512
TYPE_VOCAB_SIZE = 2
LAYER_NORM_EPS = 1e-12
MODEL_ID = "onnx-community/paraphrase-multilingual-MiniLM-L12-v2-ONNX"
MODEL_PATH = Path(__file__).parents[1] / "weights" / "model_int8.onnx"
DEFAULT_TEXT = "Tôi đang làm đồ án FPGA."


def _dynamic_quantize(x):
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    lower = torch.minimum(x.min(), zero)
    upper = torch.maximum(x.max(), zero)
    scale = (upper - lower) / 255
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero_point = torch.round(torch.clamp(-lower / scale, 0, 255))
    quantized = torch.clamp(torch.round(x / scale) + zero_point, 0, 255)
    return quantized.to(torch.int32), scale, zero_point.to(torch.int32)


class DynamicQuantizedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.register_buffer(
            "weight",
            torch.empty(in_features, out_features, dtype=torch.int8),
        )
        self.register_buffer("weight_scale", torch.empty((), dtype=torch.float32))
        self.register_buffer("weight_zero_point", torch.empty((), dtype=torch.int8))
        self.bias = nn.Parameter(torch.empty(out_features))

    def forward(self, x):
        quantized, scale, zero_point = _dynamic_quantize(x)
        accumulator = (quantized - zero_point) @ (
            self.weight.to(torch.int32) - self.weight_zero_point.to(torch.int32)
        )
        return accumulator.to(torch.float32) * (scale * self.weight_scale) + self.bias



class BertEmbeddings(nn.Module):

    def __init__(self):
        super().__init__()

        self.word_embeddings = nn.Embedding(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            padding_idx=0,
        )

        self.position_embeddings = nn.Embedding(
            MAX_POSITION,
            HIDDEN_SIZE,
        )

        self.token_type_embeddings = nn.Embedding(
            TYPE_VOCAB_SIZE,
            HIDDEN_SIZE,
        )

        self.layer_norm = nn.LayerNorm(
            HIDDEN_SIZE,
            eps=LAYER_NORM_EPS,
        )

    def forward(self, input_ids, token_type_ids=None):

        batch_size, seq_len = input_ids.shape

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_ids = torch.arange(
            seq_len,
            device=input_ids.device,
        )

        position_ids = position_ids.unsqueeze(0)

        word = self.word_embeddings(input_ids)

        position = self.position_embeddings(position_ids)

        token_type = self.token_type_embeddings(token_type_ids)

        x = word + position + token_type

        return self.layer_norm(x)



class SelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.query = DynamicQuantizedLinear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
        )

        self.key = DynamicQuantizedLinear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
        )

        self.value = DynamicQuantizedLinear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
        )

    def split_heads(self, x):

        batch_size, seq_len, _ = x.shape

        x = x.view(
            batch_size,  # 1
            seq_len,  # max 128
            NUM_HEADS,  # 12
            HEAD_DIM,  # 32
        )

        return x.transpose(1, 2) # (1, 12, max 128, 32)

    def forward(self, x, attention_mask=None):

        Q = self.split_heads(self.query(x))
        K = self.split_heads(self.key(x))
        V = self.split_heads(self.value(x)) # (1, 12, max 128, 32)

        scores = Q @ K.transpose(-2, -1) # (1, 12, max 128, max 128)

        scores = scores / math.sqrt(HEAD_DIM)

        if attention_mask is not None:

            mask = attention_mask[:, None, None, :]

            scores = scores.masked_fill(
                mask == 0,
                torch.finfo(scores.dtype).min,
            )

        probabilities = torch.softmax(
            scores,
            dim=-1,
        )

        context = probabilities @ V # (1, 12, max 128, 32)

        context = context.transpose(1, 2) # (1, max 128, 12, 32)

        batch_size, seq_len, _, _ = context.shape

        context = context.reshape(
            batch_size,
            seq_len,
            HIDDEN_SIZE, # 384
        )

        return context


class AttentionOutput(nn.Module):

    def __init__(self):
        super().__init__()

        self.dense = DynamicQuantizedLinear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
        )

        self.layer_norm = nn.LayerNorm(
            HIDDEN_SIZE,
            eps=LAYER_NORM_EPS,
        )

    def forward(self, attention, residual):

        x = self.dense(attention)

        x = x + residual

        return self.layer_norm(x)



class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.dense1 = DynamicQuantizedLinear(
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
        )

        self.dense2 = DynamicQuantizedLinear(
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
        )

        self.layer_norm = nn.LayerNorm(
            HIDDEN_SIZE,
            eps=LAYER_NORM_EPS,
        )

        self.gelu = nn.GELU()

    def forward(self, x):

        residual = x

        x = self.dense1(x)

        x = self.gelu(x)

        x = self.dense2(x)

        x = x + residual

        return self.layer_norm(x)



class EncoderLayer(nn.Module):

    def __init__(self):
        super().__init__()

        self.self_attention = SelfAttention()

        self.attention_output = AttentionOutput()

        self.ffn = FeedForward()

    def forward(self, x, attention_mask=None):

        attention = self.self_attention(
            x,
            attention_mask,
        )

        x = self.attention_output(
            attention,
            x,
        )

        x = self.ffn(x)

        return x



class MiniLM(nn.Module):

    def __init__(self):
        super().__init__()

        self.embeddings = BertEmbeddings()

        self.layers = nn.ModuleList(
            [
                EncoderLayer()
                for _ in range(NUM_LAYERS)
            ]
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
    ):

        x = self.embeddings(
            input_ids,
            token_type_ids,
        )

        for layer in self.layers:

            x = layer(
                x,
                attention_mask,
            )

        return x


def _dequantize(tensors, quantized_name, zero_point_name):
    scale_name = quantized_name.removesuffix("_quantized") + "_scale"
    return (
        tensors[quantized_name].astype(np.float32)
        - tensors[zero_point_name].astype(np.float32)
    ) * tensors[scale_name].astype(np.float32)


def _linear_parameter_name(node_name):
    parts = node_name.strip("/").removesuffix("/MatMul_quant").split("/")
    layer = parts[1].removeprefix("layer.")
    module = "/".join(parts[2:])
    modules = {
        "attention/self/query": "self_attention.query",
        "attention/self/key": "self_attention.key",
        "attention/self/value": "self_attention.value",
        "attention/output/dense": "attention_output.dense",
        "intermediate/dense": "ffn.dense1",
        "output/dense": "ffn.dense2",
    }
    return f"layers.{layer}.{modules[module]}.weight"


def _float_parameter_name(onnx_name):
    embedding_names = {
        "embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
        "embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",
    }
    if onnx_name in embedding_names:
        return embedding_names[onnx_name]

    parts = onnx_name.split(".")
    layer = parts[2]
    module = ".".join(parts[3:])
    modules = {
        "attention.self.query.bias": "self_attention.query.bias",
        "attention.self.key.bias": "self_attention.key.bias",
        "attention.self.value.bias": "self_attention.value.bias",
        "attention.output.dense.bias": "attention_output.dense.bias",
        "attention.output.LayerNorm.weight": "attention_output.layer_norm.weight",
        "attention.output.LayerNorm.bias": "attention_output.layer_norm.bias",
        "intermediate.dense.bias": "ffn.dense1.bias",
        "output.dense.bias": "ffn.dense2.bias",
        "output.LayerNorm.weight": "ffn.layer_norm.weight",
        "output.LayerNorm.bias": "ffn.layer_norm.bias",
    }
    return f"layers.{layer}.{modules[module]}"


def load_model_from_onnx(path: str | Path):
    graph = onnx.load(str(path), load_external_data=False).graph
    tensors = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }
    state_dict = {}

    for name in ("word_embeddings", "position_embeddings", "token_type_embeddings"):
        prefix = f"embeddings.{name}.weight"
        value = _dequantize(
            tensors,
            f"{prefix}_quantized",
            f"{prefix}_zero_point",
        )
        state_dict[f"embeddings.{name}.weight"] = torch.from_numpy(value.copy())

    for initializer in graph.initializer:
        if initializer.data_type != onnx.TensorProto.FLOAT:
            continue
        if initializer.name.endswith("_scale"):
            continue
        target = _float_parameter_name(initializer.name)
        state_dict[target] = torch.from_numpy(tensors[initializer.name].copy())

    for node in graph.node:
        if node.op_type != "MatMulInteger":
            continue
        prefix = _linear_parameter_name(node.name).removesuffix(".weight")
        scale_name = node.input[1].removesuffix("_quantized") + "_scale"
        state_dict[f"{prefix}.weight"] = torch.from_numpy(tensors[node.input[1]].copy())
        state_dict[f"{prefix}.weight_scale"] = torch.from_numpy(
            tensors[scale_name].copy()
        )
        state_dict[f"{prefix}.weight_zero_point"] = torch.from_numpy(
            tensors[node.input[3]].copy()
        )

    with torch.device("meta"):
        model = MiniLM()
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.eval()


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])

    model = load_model_from_onnx(MODEL_PATH)
    with torch.inference_mode():
        last_hidden_state = model(**inputs)

    print("text:", text)
    print("input_ids:", tuple(inputs["input_ids"].shape))
    print("last_hidden_state:", tuple(last_hidden_state.shape))
    print("first_token[:8]:", last_hidden_state[0, 0, :8].numpy())


if __name__ == "__main__":
    main()
