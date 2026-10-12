class SelfAttention:
    def __init__(self, tensors, layer_index):
        self.tensors = tensors
        self.layer_index = layer_index

    def __call__(self, hidden_states, attention_mask=None):
        """
        hidden_states: (batch_size, sequence_length, hidden_size)
        attention_mask: (batch_size, sequence_length)

        Returns: (batch_size, sequence_length, hidden_size)
        """
        raise NotImplementedError
