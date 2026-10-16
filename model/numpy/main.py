import sys
from transformers import AutoTokenizer

if __package__:
    from .config import DEFAULT_TEXT, MODEL_ID, MODEL_PATH
    from .model import SentenceEmbeddingModel
else:
    from config import DEFAULT_TEXT, MODEL_ID, MODEL_PATH
    from model import SentenceEmbeddingModel


def main():
    text = input("Enter text (or press Enter to use default): ") or DEFAULT_TEXT

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=128
    )

    model = SentenceEmbeddingModel(MODEL_PATH)
    embedding_output, scale_exponent = model(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        token_type_ids=inputs.get("token_type_ids")
    )

    print("Bert embedding shape:", embedding_output.shape)
    print("Bert embedding dtype:", embedding_output.dtype)
    print("Bert embedding scale exponent:", scale_exponent)
    print("Real value = int8 code * 2**scale_exponent; zero_point = 0")
    print("Bert embedding:", embedding_output)

if __name__ == "__main__":
    main()
