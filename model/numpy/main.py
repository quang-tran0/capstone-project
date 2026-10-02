import sys

from transformers import AutoTokenizer

if __package__:
    from .config import DEFAULT_TEXT, MODEL_ID, MODEL_PATH
    from .embeddings import word_embedding
    from .weights import load_initializers
else:
    from config import DEFAULT_TEXT, MODEL_ID, MODEL_PATH
    from embeddings import word_embedding
    from weights import load_initializers


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"]

    tensors = load_initializers(MODEL_PATH)
    quantized, centered, embedding = word_embedding(input_ids, tensors)

    print("text:", text)
    print("input_ids:", input_ids)

    # for name, value in (
    #     ("quantized", quantized),
    #     ("centered", centered),
    #     ("word_embedding", embedding),
    # ):
    #     print(f"\n{name}: shape={value.shape}, dtype={value.dtype}")
    #     print("first token[:8]:", value[0, 0])


if __name__ == "__main__":
    main()
