import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnx
from onnx import numpy_helper

from .config import MODEL_PATH
from .weights import _validate_weights, load_weights


class WeightLoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = load_weights(MODEL_PATH)

    def test_loads_dequantized_embedding_weights(self):
        word_embeddings = self.weights["embeddings.word_embeddings.weight"]

        self.assertEqual(word_embeddings.shape, (250037, 384))
        self.assertEqual(word_embeddings.dtype, np.float32)

    def test_keeps_linear_weights_quantized(self):
        prefix = "layers.0.self_attention.query"

        self.assertEqual(self.weights[f"{prefix}.weight"].shape, (384, 384))
        self.assertEqual(self.weights[f"{prefix}.weight"].dtype, np.int8)
        self.assertEqual(self.weights[f"{prefix}.weight_scale"].shape, ())
        self.assertEqual(self.weights[f"{prefix}.weight_scale"].dtype, np.float32)
        self.assertEqual(self.weights[f"{prefix}.weight_zero_point"].shape, ())
        self.assertEqual(self.weights[f"{prefix}.weight_zero_point"].dtype, np.int8)
        self.assertEqual(self.weights[f"{prefix}.bias"].shape, (384,))

    def test_loads_all_encoder_layers(self):
        for layer in range(12):
            self.assertIn(
                f"layers.{layer}.self_attention.query.weight",
                self.weights,
            )

    def test_validation_reports_missing_tensor(self):
        weights = self.weights.copy()
        del weights["embeddings.word_embeddings.weight"]

        with self.assertRaisesRegex(
            ValueError,
            r"missing.*embeddings\.word_embeddings\.weight",
        ):
            _validate_weights(weights)

    def test_loader_rejects_unknown_initializers(self):
        source = onnx.load(str(MODEL_PATH), load_external_data=False)
        unknown_initializers = (
            ("unexpected_quantized", np.array([1], dtype=np.int8)),
            ("unexpected_scale", np.array(1.0, dtype=np.float32)),
            (
                "encoder.layer.12.output.LayerNorm.bias",
                np.zeros(384, dtype=np.float32),
            ),
        )

        for name, value in unknown_initializers:
            graph = copy.deepcopy(source.graph)
            graph.initializer.append(numpy_helper.from_array(value, name=name))
            with self.subTest(name=name), patch(
                "model.numpy.weights.onnx.load",
                return_value=SimpleNamespace(graph=graph),
            ):
                with self.assertRaisesRegex(ValueError, r"unexpected.*initializer"):
                    load_weights(MODEL_PATH)


if __name__ == "__main__":
    unittest.main()
