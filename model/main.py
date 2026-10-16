import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.numpy.model import SentenceEmbeddingModel as NumpyModel
from model.pytorch.main import MODEL_PATH as PYTORCH_MODEL_PATH, load_model_from_onnx


MODEL_ID = "onnx-community/paraphrase-multilingual-MiniLM-L12-v2-ONNX"
MODEL_PATH = Path(__file__).parent / "weights" / "model_int8.onnx"
DEFAULT_TEXT = "Tôi đang làm đồ án FPGA."


def percentage_errors(reference, actual, eps=1e-8):
    """Mean and maximum absolute relative error, with PyTorch as reference."""
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape:
        raise ValueError(f"Shape mismatch: PyTorch {reference.shape}, NumPy {actual.shape}")
    relative = 100 * np.abs(actual - reference) / np.maximum(np.abs(reference), eps)
    return float(relative.mean()), float(relative.max())


def trace_pytorch_embeddings(embeddings, input_ids, token_type_ids=None):
    """Capture the same five stages as the NumPy BertEmbeddings trace."""
    trace = {}
    handles = []

    def save_output(name):
        def hook(module, args, output):
            trace[name] = output.detach().cpu().numpy().copy()
        return hook

    def save_sum(module, args):
        trace["embeddings.sum"] = args[0].detach().cpu().numpy().copy()

    try:
        for name, module in (
            ("embeddings.word", embeddings.word_embeddings),
            ("embeddings.position", embeddings.position_embeddings),
            ("embeddings.token_type", embeddings.token_type_embeddings),
            ("embeddings.output", embeddings.layer_norm),
        ):
            handles.append(module.register_forward_hook(save_output(name)))
        handles.append(embeddings.layer_norm.register_forward_pre_hook(save_sum))
        with torch.inference_mode():
            embeddings(input_ids, token_type_ids)
    finally:
        for handle in handles:
            handle.remove()
    return trace


def symmetric_quantize_pytorch(x):
    """PyTorch reference for the current per-tensor symmetric INT8 quantizer."""
    scale = x.abs().max() / 127
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return quantized, float(scale.item())


def main():
    text = input("Enter text (or press Enter to use default): ") or DEFAULT_TEXT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=128,
    )
    inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])

    numpy_model = NumpyModel(MODEL_PATH, trace=True)
    quantized, scale = numpy_model(**inputs)
    pytorch_model = load_model_from_onnx(PYTORCH_MODEL_PATH)
    pytorch_trace = trace_pytorch_embeddings(
        pytorch_model.embeddings,
        torch.from_numpy(inputs["input_ids"]),
        torch.from_numpy(inputs["token_type_ids"]),
    )
    pytorch_quantized, pytorch_scale = symmetric_quantize_pytorch(
        torch.from_numpy(pytorch_trace["embeddings.output"])
    )
    pytorch_trace["embeddings.quantized"] = pytorch_quantized.numpy()

    print("text:", text)
    print("input_ids:", inputs["input_ids"].shape)
    print(f"NumPy weights: {MODEL_PATH.name}; PyTorch float32 reference: {PYTORCH_MODEL_PATH.name}")
    print("Both models use the same tokenized inputs.")
    print("Comparison quantizer: per-tensor symmetric int8, zero_point=0.")
    print("Error (%) = 100 * abs(NumPy - PyTorch) / max(abs(PyTorch), 1e-8)")
    print(f"{'Layer':<24} {'Shape':<20} {'Mean error (%)':>18} {'Max error (%)':>18}")
    print("-" * 83)
    for name in (
        "embeddings.word",
        "embeddings.position",
        "embeddings.token_type",
        "embeddings.sum",
        "embeddings.output",
        "embeddings.quantized",
    ):
        reference = pytorch_trace[name]
        mean_error, max_error = percentage_errors(reference, numpy_model.layers_output[name])
        print(f"{name:<24} {str(reference.shape):<20} {mean_error:18.8f} {max_error:18.8f}")

    restored = quantized.astype(np.float64) * scale
    reference = pytorch_trace["embeddings.output"]
    mean_error, max_error = percentage_errors(reference, restored)
    print(f"{'embeddings.dequantized':<24} {str(restored.shape):<20} {mean_error:18.8f} {max_error:18.8f}")
    print(f"NumPy int8 output: scale={scale:.9g}, zero_point=0")
    print(f"PyTorch comparison quantizer: scale={pytorch_scale:.9g}, zero_point=0")


if __name__ == "__main__":
    main()
