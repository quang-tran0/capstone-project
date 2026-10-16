class SelfAttention:
    def __init__(self, tensors, trace=False, layers_output=None):
        self.tensors = tensors
        self.trace = trace
        self.layers_output = layers_output

    def __call__(self, hidden_states, attention_mask=None):
        """
        hidden_states: (batch_size, sequence_length, hidden_size)
        attention_mask: (batch_size, sequence_length)

        Returns: (batch_size, sequence_length, hidden_size)
        """
        raise NotImplementedError
