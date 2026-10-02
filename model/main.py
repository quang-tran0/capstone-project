import os
import sys
from pathlib import Path

import numpy as np
import onnx


os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import onnxruntime as ort
from transformers import AutoTokenizer


MODEL_ID = "onnx-community/paraphrase-multilingual-MiniLM-L12-v2-ONNX"
MODEL_PATH = Path(__file__).parent / "weights" / "model_int8.onnx"
DEFAULT_TEXT = "Tôi đang làm đồ án FPGA."


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=128,
    )
    inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )
    model_inputs = {item.name: inputs[item.name] for item in session.get_inputs()}
    last_hidden_state = session.run(None, model_inputs)[0]

    print("text:", text)
    print("input_ids:", inputs["input_ids"].shape)
    print("last_hidden_state:", last_hidden_state.shape)
    print("first_token[:8]:", last_hidden_state[0, 0, :8])

    model = onnx.load("weights/model_int8.onnx")

    for init in model.graph.initializer:
        name = init.name.lower()

        if "embedding" in name:
            print(
                init.name,
                "dtype =", init.data_type,
                "shape =", init.dims,
            )

if __name__ == "__main__":
    main()
