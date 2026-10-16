import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from model.pytorch.main import SentenceEmbeddingModel, load_model_from_onnx


MODEL_PATH = Path(__file__).parent / "weights" / "model.onnx"


class PytorchFloatTest(unittest.TestCase):
    def test_comparison_quantizer_matches_current_numpy_scale_contract(self):
        from model.main import symmetric_quantize_pytorch

        quantized, scale = symmetric_quantize_pytorch(
            torch.tensor([[-2., -1., 0., 1., 3.]])
        )
        np.testing.assert_array_equal(quantized.numpy(), [[-85, -42, 0, 42, 127]])
        self.assertEqual(scale, float(np.float32(3) / np.float32(127)))
        zero_quantized, zero_scale = symmetric_quantize_pytorch(torch.zeros(1, 3))
        np.testing.assert_array_equal(zero_quantized.numpy(), [[0, 0, 0]])
        self.assertEqual(zero_scale, 1.0)

    def test_float_encoder_and_pooling_match_onnx_with_padding(self):
        encoder = load_model_from_onnx(MODEL_PATH)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in encoder.parameters()))
        self.assertEqual(tuple(encoder.layers[0].ffn.dense1.weight.shape), (1536, 384))
        self.assertEqual(tuple(encoder.layers[0].ffn.dense2.weight.shape), (384, 1536))
        inputs = {
            "input_ids": np.array(
                [[0, 87, 262, 9384, 2, 1], [0, 14343, 4724, 1839, 15091, 2]],
                dtype=np.int64,
            ),
            "attention_mask": np.array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]], dtype=np.int64),
            "token_type_ids": np.array([[0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 1, 1]], dtype=np.int64),
        }
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(
            str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"]
        )
        expected = session.run(None, inputs)[0]
        torch_inputs = {name: torch.from_numpy(value) for name, value in inputs.items()}
        with torch.inference_mode():
            actual = encoder(**torch_inputs).numpy()
            pooled = SentenceEmbeddingModel(encoder).eval()(**torch_inputs).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        expected_pooled = np.stack([expected[0, :5].mean(axis=0), expected[1].mean(axis=0)])
        self.assertEqual(pooled.shape, (2, 384))
        np.testing.assert_allclose(pooled, expected_pooled, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
