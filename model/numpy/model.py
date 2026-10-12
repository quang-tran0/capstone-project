if __package__:
    from .weights import load_initializers
    from .layers.bert_embedding import BertEmbeddings
else:
    from weights import load_initializers
    from layers.bert_embedding import BertEmbeddings

class SentenceEmbeddingModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tensors = load_initializers(model_path)
        self.bert_embeddings = BertEmbeddings(self.tensors)

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        input_ids: (batch_size, sequence_length)
        attention_mask: (batch_size, sequence_length)
        token_type_ids: (batch_size, sequence_length)

        Returns: (batch_size, 384)
        """

        x = self.bert_embeddings(
            input_ids,
            token_type_ids
        ) # (batch_size, sequence_length, 384)


        return x
