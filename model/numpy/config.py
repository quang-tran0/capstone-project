from pathlib import Path

MODEL_ID = "onnx-community/paraphrase-multilingual-MiniLM-L12-v2-ONNX"
MODEL_PATH = Path(__file__).resolve().parents[1] / "weights" / "model_int8.onnx"
DEFAULT_TEXT = "AXI hỗ trợ nhiều giao dịch outstanding như thế nào?"
