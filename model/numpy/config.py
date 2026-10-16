from pathlib import Path

MODEL_ID = "onnx-community/paraphrase-multilingual-MiniLM-L12-v2-ONNX"
MODEL_PATH = Path(__file__).resolve().parents[1] / "weights" / "model.onnx"
DEFAULT_TEXT = "AXI hỗ trợ nhiều giao dịch outstanding như thế nào?"

HIDDEN_SIZE = 384
NUM_HEADS = 12
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS # 384 / 12 = 32
INTERMEDIATE_SIZE = 1536
NUM_LAYERS = 12
VOCAB_SIZE = 250037
MAX_POSITION = 512
TYPE_VOCAB_SIZE = 2
LAYER_NORM_EPS = 1e-12
