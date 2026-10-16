import unittest

import numpy as np
import torch

from model.main import percentage_errors, trace_pytorch_embeddings
from model.numpy.config import MODEL_PATH
from model.numpy.model import SentenceEmbeddingModel
from model.pytorch.main import MODEL_PATH as TORCH_MODEL_PATH, load_model_from_onnx


class NumpyEmbeddingsTest(unittest.TestCase):
    def test_percentage_errors_use_the_full_reference_range(self):
        reference = np.array([[-2., 0.], [2., 2.]])
        actual = np.array([[-1., 0.], [4., 2.]])
        np.testing.assert_allclose(percentage_errors(reference, actual), (18.75, 50.0))
        self.assertEqual(percentage_errors(np.ones(3), np.ones(3)), (0.0, 0.0))
        with self.assertRaises(ValueError):
            percentage_errors(np.zeros((1, 3)), np.zeros(3))

    def test_float_embeddings_and_dequantized_int8_trace(self):
        self.assertEqual(MODEL_PATH, TORCH_MODEL_PATH)
        model = SentenceEmbeddingModel(MODEL_PATH, trace=True)
        reference = load_model_from_onnx(TORCH_MODEL_PATH).embeddings
        ids = np.array(
            [[0, 87, 262, 9384, 2, 1], [0, 14343, 4724, 1839, 15091, 2]],
            dtype=np.int64,
        )
        mask = np.array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]], dtype=np.int64)
        cases = (None, np.array([[0, 0, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0]], dtype=np.int64))
        for types in cases:
            with self.subTest(token_type_ids=types):
                output, scale = model(ids, attention_mask=mask, token_type_ids=types)
                expected = trace_pytorch_embeddings(
                    reference, torch.from_numpy(ids),
                    None if types is None else torch.from_numpy(types),
                )
                self.assertIsInstance(output, np.ndarray)
                self.assertEqual(output.shape, (2, 6, 384))
                self.assertEqual(output.dtype, np.int8)
                self.assertGreater(scale, 0)
                raw = model.bert_embeddings.layer_norm(model.layers_output["embeddings.sum"])
                for name, values in expected.items():
                    actual = raw if name == "embeddings.output" else model.layers_output[name]
                    np.testing.assert_allclose(
                        actual, values, rtol=1e-5, atol=2e-6,
                        err_msg=name,
                    )
                self.assertEqual(raw.dtype, np.float32)
                self.assertAlmostEqual(scale, float(np.abs(expected["embeddings.output"]).max() / 128), places=7)
                np.testing.assert_array_equal(model.layers_output["embeddings.quantized"], output)
                self.assertEqual(model.layers_output["embeddings.scale"], scale)
                restored = output.astype(np.float32) * scale
                np.testing.assert_array_equal(model.layers_output["embeddings.output"], restored)
                # The +128 endpoint clips to +127 in the existing quantizer.
                self.assertLessEqual(float(np.abs(restored - raw).max()), scale + 1e-6)


if __name__ == "__main__":
    unittest.main()
