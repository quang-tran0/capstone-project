import math
import sys
from pathlib import Path

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
MODEL_PATH = Path(__file__).resolve().parents[1] / "weights" / "model.onnx"
DEFAULT_TEXT = "Tôi đang làm đồ án FPGA."


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

        self.query = nn.Linear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
        )

        self.key = nn.Linear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
        )

        self.value = nn.Linear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
        )

    def split_heads(self, x):

        batch_size, seq_len, _ = x.shape

        x = x.view(
            batch_size,  # 1
            seq_len,  # seq length
            NUM_HEADS,  # 12
            HEAD_DIM,  # 32
        )

        return x.transpose(1, 2) # (1, 12, seq length, 32)

    def forward(self, x, attention_mask=None):
        # x: (1, seq length, 384)
        Q = self.split_heads(self.query(x))
        K = self.split_heads(self.key(x))
        V = self.split_heads(self.value(x)) # (1, 12, seq length, 32)

        scores = Q @ K.transpose(-2, -1) # (1, 12, seq length, seq length)

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

        context = probabilities @ V # (1, 12, seq length, 32)

        context = context.transpose(1, 2) # (1, seq length, 12, 32)

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

        self.dense = nn.Linear(
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

        self.dense1 = nn.Linear(
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
        )

        self.dense2 = nn.Linear(
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


class MeanPooling(nn.Module):
    """Average token embeddings, excluding padding: (B, N, 384) -> (B, 384)."""

    def forward(self, last_hidden_state, attention_mask=None):
        if attention_mask is None:
            return last_hidden_state.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        return summed / mask.sum(dim=1).clamp(min=1)


class SentenceEmbeddingModel(nn.Module):
    """MiniLM encoder followed by mean pooling into one vector per sentence."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.pooling = MeanPooling()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        last_hidden_state = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return self.pooling(last_hidden_state, attention_mask)


def _linear_parameter_name(node_name):
    parts = node_name.strip("/").removesuffix("/MatMul").split("/")
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
    if onnx_name.startswith("embeddings."):
        return onnx_name.replace("embeddings.LayerNorm.", "embeddings.layer_norm.")

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
    """Load the float32 ONNX encoder, transposing MatMul weights for nn.Linear."""
    graph = onnx.load(str(path)).graph
    tensors = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }
    state_dict = {}

    linear_weights = {}
    for node in graph.node:
        if node.op_type == "MatMul" and node.input[1] in tensors:
            linear_weights[node.input[1]] = _linear_parameter_name(node.name)

    for initializer in graph.initializer:
        if initializer.data_type != onnx.TensorProto.FLOAT:
            continue
        if initializer.name in linear_weights:
            continue
        target = _float_parameter_name(initializer.name)
        state_dict[target] = torch.from_numpy(tensors[initializer.name].copy())

    for name, target in linear_weights.items():
        state_dict[target] = torch.from_numpy(tensors[name].T.copy())

    with torch.device("meta"):
        model = MiniLM()
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.eval()


def main():
    text = input("Enter text (or press Enter to use default): ") or DEFAULT_TEXT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])

    model = SentenceEmbeddingModel(load_model_from_onnx(MODEL_PATH)).eval()
    with torch.inference_mode():
        sentence_embedding = model(**inputs)

    print("text:", text)
    print("input_ids:", tuple(inputs["input_ids"].shape))
    print("sentence_embedding:", tuple(sentence_embedding.shape))
    print("sentence_embedding[:8]:", sentence_embedding[0, :8].numpy())


if __name__ == "__main__":
    main()
