import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.numpy.config import MODEL_PATH
from model.numpy.model import SentenceEmbeddingModel as NumpyModel
from model.pytorch.main import MODEL_PATH as PYTORCH_MODEL_PATH, load_model_from_onnx


MODEL_ID = "onnx-community/paraphrase-multilingual-MiniLM-L12-v2-ONNX"
DEFAULT_TEXT = "Tôi đang làm đồ án FPGA."


def percentage_errors(reference, actual, eps=1e-8):
    """Mean and maximum error normalized by the full PyTorch tensor's range."""
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape:
        raise ValueError(f"Shape mismatch: PyTorch {reference.shape}, NumPy {actual.shape}")
    denominator = max(float(reference.max() - reference.min()), eps)
    relative = 100 * np.abs(actual - reference) / denominator
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
    numpy_model(**inputs)
    pytorch_model = load_model_from_onnx(PYTORCH_MODEL_PATH)
    pytorch_trace = trace_pytorch_embeddings(
        pytorch_model.embeddings,
        torch.from_numpy(inputs["input_ids"]),
        torch.from_numpy(inputs["token_type_ids"]),
    )

    print("text:", text)
    print("input_ids:", inputs["input_ids"].shape)
    print(f"NumPy weights: {MODEL_PATH.name}; PyTorch float32 reference: {PYTORCH_MODEL_PATH.name}")
    print("Both models use the same tokenized inputs.")
    print("Error (%) = 100 * abs(NumPy - PyTorch) / max(max(PyTorch) - min(PyTorch), 1e-8)")
    print(f"{'Layer':<24} {'Shape':<20} {'Mean error (%)':>18} {'Max error (%)':>18} {'MAE':>12} {'Max abs':>12}")
    print("-" * 109)
    for name in (
        "embeddings.word",
        "embeddings.position",
        "embeddings.token_type",
        "embeddings.sum",
        "embeddings.output",
    ):
        reference = pytorch_trace[name]
        mean_error, max_error = percentage_errors(reference, numpy_model.layers_output[name])
        absolute = np.abs(numpy_model.layers_output[name].astype(np.float64) - reference)
        print(f"{name:<24} {str(reference.shape):<20} {mean_error:18.8f} {max_error:18.8f} {absolute.mean():12.6g} {absolute.max():12.6g}")

    print("PyTorch max/min are computed over the full tensor of each layer.")


if __name__ == "__main__":
    main()
