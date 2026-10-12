import numpy as np

class BertEmbeddings:
    def __init__(self, tensors):
        self.tensors = tensors

        prefix = "embeddings.word_embeddings"
        self.word_embeddings = tensors[f"{prefix}.weight_quantized"] # (vocab_size, hidden_size)
        self.word_zero_point = int(tensors[f"{prefix}.weight_zero_point"])
        self.word_scale = tensors[f"{prefix}.weight_scale"]

    def forward(self, input_ids, token_type_ids=None):
        # 1. Tra bảng word embedding.
        # 2. Tra bảng position embedding.
        # 3. Tra bảng token type embedding.
        # 4. Đưa về cùng thang số rồi cộng.
        # 5. LayerNorm.
        raise NotImplementedError
