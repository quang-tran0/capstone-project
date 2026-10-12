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
    sentence_embedding = model(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        token_type_ids=inputs.get("token_type_ids")
    )

    print("Sentence embedding shape:", sentence_embedding.shape)

if __name__ == "__main__":
    main()
