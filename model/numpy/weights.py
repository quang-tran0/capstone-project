from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from .config import NUM_LAYERS


LINEAR_MODULES = (
    "self_attention.query",
    "self_attention.key",
    "self_attention.value",
    "attention_output.dense",
    "ffn.dense1",
    "ffn.dense2",
)


def _dequantize(tensors, quantized_name, zero_point_name):
    scale_name = quantized_name.removesuffix("_quantized") + "_scale"
    return (
        tensors[quantized_name].astype(np.float32)
        - tensors[zero_point_name].astype(np.float32)
    ) * tensors[scale_name].astype(np.float32)


def _linear_name(node_name):
    parts = node_name.strip("/").removesuffix("/MatMul_quant").split("/")
    if len(parts) < 4 or parts[0] != "encoder" or not parts[1].startswith("layer."):
        raise ValueError(f"unsupported MatMulInteger node: {node_name}")

    modules = {
        "attention/self/query": "self_attention.query",
        "attention/self/key": "self_attention.key",
        "attention/self/value": "self_attention.value",
        "attention/output/dense": "attention_output.dense",
        "intermediate/dense": "ffn.dense1",
        "output/dense": "ffn.dense2",
    }
    source_module = "/".join(parts[2:])
    if source_module not in modules:
        raise ValueError(f"unsupported MatMulInteger node: {node_name}")

    layer = parts[1].removeprefix("layer.")
    return f"layers.{layer}.{modules[source_module]}"


def _float_name(initializer_name):
    embedding_names = {
        "embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
        "embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",
    }
    if initializer_name in embedding_names:
        return embedding_names[initializer_name]

    parts = initializer_name.split(".")
    if len(parts) < 5 or parts[:2] != ["encoder", "layer"]:
        raise ValueError(f"unsupported float initializer: {initializer_name}")

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
    source_module = ".".join(parts[3:])
    if source_module not in modules:
        raise ValueError(f"unsupported float initializer: {initializer_name}")

    return f"layers.{parts[2]}.{modules[source_module]}"


def _expected_weight_names():
    names = {
        "embeddings.word_embeddings.weight",
        "embeddings.position_embeddings.weight",
        "embeddings.token_type_embeddings.weight",
        "embeddings.layer_norm.weight",
        "embeddings.layer_norm.bias",
    }
    for layer in range(NUM_LAYERS):
        for module in LINEAR_MODULES:
            prefix = f"layers.{layer}.{module}"
            names.update(
                {
                    f"{prefix}.weight",
                    f"{prefix}.weight_scale",
                    f"{prefix}.weight_zero_point",
                    f"{prefix}.bias",
                }
            )
        names.update(
            {
                f"layers.{layer}.attention_output.layer_norm.weight",
                f"layers.{layer}.attention_output.layer_norm.bias",
                f"layers.{layer}.ffn.layer_norm.weight",
                f"layers.{layer}.ffn.layer_norm.bias",
            }
        )
    return names


def _validate_weights(weights):
    expected = _expected_weight_names()
    missing = sorted(expected - weights.keys())
    if missing:
        raise ValueError(f"missing expected tensor: {missing[0]}")
    unexpected = sorted(weights.keys() - expected)
    if unexpected:
        raise ValueError(
            f"unexpected tensor derived from initializer: {unexpected[0]}"
        )


def _validate_initializers(tensors, consumed):
    unexpected = sorted(tensors.keys() - consumed)
    if unexpected:
        raise ValueError(f"unexpected ONNX initializer: {unexpected[0]}")


def load_weights(path: str | Path):
    graph = onnx.load(str(path), load_external_data=False).graph
    tensors = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }
    weights = {}
    consumed = set()

    for name in ("word_embeddings", "position_embeddings", "token_type_embeddings"):
        prefix = f"embeddings.{name}.weight"
        quantized_name = f"{prefix}_quantized"
        zero_point_name = f"{prefix}_zero_point"
        scale_name = f"{prefix}_scale"
        weights[prefix] = _dequantize(
            tensors,
            quantized_name,
            zero_point_name,
        ).astype(np.float32, copy=False)
        consumed.update((quantized_name, zero_point_name, scale_name))

    for initializer in graph.initializer:
        if initializer.data_type != onnx.TensorProto.FLOAT:
            continue
        if initializer.name.endswith("_scale"):
            continue
        weights[_float_name(initializer.name)] = tensors[initializer.name].copy()
        consumed.add(initializer.name)

    for node in graph.node:
        if node.op_type != "MatMulInteger":
            continue
        prefix = _linear_name(node.name)
        scale_name = node.input[1].removesuffix("_quantized") + "_scale"
        weights[f"{prefix}.weight"] = tensors[node.input[1]].copy()
        weights[f"{prefix}.weight_scale"] = tensors[scale_name].copy()
        weights[f"{prefix}.weight_zero_point"] = tensors[node.input[3]].copy()
        consumed.update((node.input[1], scale_name, node.input[3]))

    _validate_initializers(tensors, consumed)
    _validate_weights(weights)
    return weights
