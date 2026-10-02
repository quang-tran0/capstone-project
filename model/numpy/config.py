from pathlib import Path


HIDDEN_SIZE = 384
NUM_HEADS = 12
HEAD_DIM = 32
INTERMEDIATE_SIZE = 1536
NUM_LAYERS = 12
MAX_POSITION = 512
LAYER_NORM_EPS = 1e-12

MODEL_ID = "onnx-community/paraphrase-multilingual-MiniLM-L12-v2-ONNX"
MODEL_PATH = Path(__file__).parents[1] / "weights" / "model_int8.onnx"
DEFAULT_TEXT = "Tôi đang làm đồ án FPGA."
