import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.numpy.config import MODEL_PATH
from model.numpy.model import SentenceEmbeddingModel as NumpyModel
from model.numpy.utils.fixed import to_float
from model.numpy.utils.similarity import SCORE_FRAC_BITS, cosine_fixed
from model.pytorch.main import (
    MODEL_PATH as PYTORCH_MODEL_PATH,
    SentenceEmbeddingModel as PytorchModel,
    load_model_from_onnx,
)


MODEL_ID = "onnx-community/paraphrase-multilingual-MiniLM-L12-v2-ONNX"
DEFAULT_TEXT = "Tôi đang làm đồ án FPGA."
DEFAULT_PASSAGES = (
    "Đồ án dùng bo mạch KV260 để tăng tốc mô hình nhúng câu.",
    "AXI hỗ trợ nhiều giao dịch outstanding nhờ ID của từng giao dịch.",
    "Hôm nay trời mưa nên tôi ở nhà đọc sách.",
    "Systolic array thực hiện phép nhân ma trận INT8 trên FPGA.",
)


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


def read_passages():
    print("Enter passages, one per line; an empty line finishes (empty first line uses defaults):")
    passages = []
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        passages.append(line)
    return passages or list(DEFAULT_PASSAGES)


def tokenize(tokenizer, text):
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=128,
    )
    inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
    return inputs


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    text = input("Enter query (or press Enter to use default): ") or DEFAULT_TEXT
    passages = read_passages()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenize(tokenizer, text)

    numpy_model = NumpyModel(MODEL_PATH, trace=True)
    pytorch_model = load_model_from_onnx(PYTORCH_MODEL_PATH)
    sentence_model = PytorchModel(pytorch_model).eval()

    # Run every sentence through both models; the query goes first.
    texts = [text, *passages]
    pytorch_outputs, numpy_fixed = [], []
    for sentence in texts:
        sentence_inputs = tokenize(tokenizer, sentence)
        with torch.inference_mode():
            pytorch_outputs.append(sentence_model(
                **{name: torch.from_numpy(value) for name, value in sentence_inputs.items()}
            ).numpy()[0])
        numpy_fixed.append(numpy_model(**sentence_inputs)[0])
    pytorch_outputs = np.stack(pytorch_outputs)
    numpy_fixed = np.stack(numpy_fixed)
    numpy_outputs = to_float(numpy_fixed)

    # Re-run the query alone so the NumPy trace matches the embedding table below.
    numpy_model(**inputs)
    pytorch_output, numpy_output = pytorch_outputs[:1], numpy_outputs[:1]
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

    # Sentence embeddings after all encoder layers and mean pooling.
    mean_error, max_error = percentage_errors(pytorch_output, numpy_output)
    absolute = np.abs(numpy_output - pytorch_output)
    cosine = np.sum(numpy_output * pytorch_output, axis=-1) / (
        np.linalg.norm(numpy_output, axis=-1) * np.linalg.norm(pytorch_output, axis=-1)
    )
    print("\n===== Sentence embedding (query) =====")
    print("PyTorch output[:8]:", pytorch_output[0, :8])
    print("NumPy output[:8]:  ", numpy_output[0, :8])
    print(f"Shape: {pytorch_output.shape}")
    print(f"Mean error (%): {mean_error:.4f}")
    print(f"Max error (%):  {max_error:.4f}")
    print(f"MAE: {absolute.mean():.6g}   Max abs: {absolute.max():.6g}")
    print(f"Cosine similarity: {cosine.min():.6f}")

    print("\n===== Sentence embeddings, all texts =====")
    print(f"{'#':<3} {'Mean error (%)':>15} {'Max error (%)':>14} {'Cosine':>10}  Text")
    for index, sentence in enumerate(texts):
        mean_error, max_error = percentage_errors(pytorch_outputs[index], numpy_outputs[index])
        print(f"{index:<3} {mean_error:15.4f} {max_error:14.4f} "
              f"{cosine_similarity(pytorch_outputs[index], numpy_outputs[index]):10.6f}  {sentence}")

    # RAG score: cosine(query, passage). NumPy uses the integer cosine in Q.15.
    pytorch_scores = np.array([cosine_similarity(pytorch_outputs[0], passage) for passage in pytorch_outputs[1:]])
    numpy_scores = cosine_fixed(numpy_fixed[:1], numpy_fixed[1:]) / (1 << SCORE_FRAC_BITS)
    pytorch_rank = np.argsort(-pytorch_scores, kind="stable")
    numpy_rank = np.argsort(-numpy_scores, kind="stable")

    print("\n===== RAG scores: cosine(query, passage) =====")
    print("Score error (%) = 100 * abs(NumPy - PyTorch) / abs(PyTorch)")
    print(f"{'#':<3} {'PyTorch':>9} {'NumPy':>9} {'Abs error':>10} {'Error (%)':>10} "
          f"{'Rank PT':>8} {'Rank NP':>8}  Passage")
    for index, passage in enumerate(passages):
        difference = abs(numpy_scores[index] - pytorch_scores[index])
        print(f"{index + 1:<3} {pytorch_scores[index]:9.5f} {numpy_scores[index]:9.5f} "
              f"{difference:10.5f} {100 * difference / max(abs(pytorch_scores[index]), 1e-8):10.4f} "
              f"{int(np.where(pytorch_rank == index)[0][0]) + 1:8d} "
              f"{int(np.where(numpy_rank == index)[0][0]) + 1:8d}  {passage}")
    errors = np.abs(numpy_scores - pytorch_scores)
    print(f"Mean abs score error: {errors.mean():.5f}   Max abs score error: {errors.max():.5f}")
    print("Ranking identical:", bool(np.array_equal(pytorch_rank, numpy_rank)))


if __name__ == "__main__":
    main()
