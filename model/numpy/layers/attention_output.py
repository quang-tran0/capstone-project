class AttentionOutput:
    def __init__(self, tensors):
        self.tensors = tensors

    def __call__(self, hidden_states, attention_output):

        raise NotImplementedError
