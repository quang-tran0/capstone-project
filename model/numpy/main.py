import argparse
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

if __package__:
    from .config import DEFAULT_TEXT, MODEL_ID, MODEL_PATH
    from .model import MiniLM
    from .trace import MarkdownTrace
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from model.numpy.config import DEFAULT_TEXT, MODEL_ID, MODEL_PATH
    from model.numpy.model import MiniLM
    from model.numpy.trace import MarkdownTrace


def main():
    parser = argparse.ArgumentParser(description="Run the NumPy MiniLM model")
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT)
    parser.add_argument(
        "--trace-dir",
        help="write trace_summary.md and trace_full.md to this directory",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenizer(
        args.text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=128,
    )
    inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])

    model = MiniLM.from_onnx(MODEL_PATH)
    if args.trace_dir:
        with MarkdownTrace(args.trace_dir) as trace:
            last_hidden_state = model(**inputs, trace=trace)
    else:
        last_hidden_state = model(**inputs)

    print("text:", args.text)
    print("input_ids:", inputs["input_ids"].shape)
    print("last_hidden_state:", last_hidden_state.shape)
    print("first_token[:8]:", last_hidden_state[0, 0, :8])
    if args.trace_dir:
        print("trace_summary:", trace.summary_path)
        print("trace_full:", trace.full_path)


if __name__ == "__main__":
    main()
