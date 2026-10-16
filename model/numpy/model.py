if __package__:
    from .weights import load_initializers
    from .layers.bert_embedding import BertEmbeddings
else:
    from weights import load_initializers
    from layers.bert_embedding import BertEmbeddings

class SentenceEmbeddingModel:
    def __init__(self, model_path, trace=False):
        self.model_path = model_path
        self.trace = trace
        self.tensors = load_initializers(model_path)

        # Save layer output for comparing with ONNX pytorch output
        self.layers_output = {}
        self.bert_embeddings = BertEmbeddings(
            self.tensors, trace=trace, layers_output=self.layers_output
        )

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        input_ids: (batch_size, sequence_length)
        attention_mask: (batch_size, sequence_length)
        token_type_ids: (batch_size, sequence_length)

        Returns: (int8 embeddings, int16 scale_exponent), zero point is zero.
        Real values are approximated by embeddings * 2**scale_exponent.
        Embeddings have shape (batch_size, sequence_length, 384).
        """
        self.layers_output.clear()

        return self.bert_embeddings(
            input_ids,
            token_type_ids,
        )
