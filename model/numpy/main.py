import sys
from pathlib import Path

from transformers import AutoTokenizer

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.numpy.config import DEFAULT_TEXT, MODEL_ID, MODEL_PATH
from model.numpy.model import SentenceEmbeddingModel


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

    model = SentenceEmbeddingModel(MODEL_PATH, trace=True)
    embedding_output, scale = model(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        token_type_ids=inputs.get("token_type_ids")
    )

    print("Bert embedding shape:", embedding_output.shape)
    print("Bert embedding dtype:", embedding_output.dtype)
    print("Bert embedding scale:", scale)
    model.display_trace()

if __name__ == "__main__":
    main()
