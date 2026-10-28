import sys

import numpy as np
from transformers import AutoTokenizer

if __package__:
    from .config import DEFAULT_TEXT, MODEL_ID, MODEL_PATH
    from .embeddings import embeddings_fixed, prepare_embeddings
    from .layernorm import (
        LAYERNORM_OUT_FRAC_BITS,
        layernorm_fixed,
        prepare_layernorm,
        prepare_rsqrt_lut,
    )
    from .weights import load_initializers
else:
    from config import DEFAULT_TEXT, MODEL_ID, MODEL_PATH
    from embeddings import embeddings_fixed, prepare_embeddings
    from layernorm import (
        LAYERNORM_OUT_FRAC_BITS,
        layernorm_fixed,
        prepare_layernorm,
        prepare_rsqrt_lut,
    )
    from weights import load_initializers


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"]
    token_type_ids = np.zeros_like(input_ids)

    # Chuẩn bị trên máy tính.
    tensors = load_initializers(MODEL_PATH)
    tables, sum_bits = prepare_embeddings(tensors)
    gamma, beta = prepare_layernorm(tensors, "embeddings.LayerNorm")
    lut = prepare_rsqrt_lut()

    # Đường tính số nguyên mô phỏng FPGA.
    x = embeddings_fixed(input_ids, token_type_ids, tables, sum_bits)
    hidden = layernorm_fixed(x, gamma, beta, lut)

    print("text:", text)
    print("input_ids:", input_ids)
    print(f"embedding_sum: shape={x.shape}, {sum_bits} bit")
    print(f"embedding_out: shape={hidden.shape}, dtype={hidden.dtype}, "
          f"Q.{LAYERNORM_OUT_FRAC_BITS}")
    print("first token[:8]:", hidden[0, 0, :8])
    print("first token[:8] (số thực):", hidden[0, 0, :8] / (1 << LAYERNORM_OUT_FRAC_BITS))


if __name__ == "__main__":
    main()
