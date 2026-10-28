import os
import unittest

import numpy as np
import onnx

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import onnxruntime as ort

from model.numpy import config
from model.numpy.embeddings import EMBED_FRAC_BITS, embeddings_fixed, prepare_embeddings
from model.numpy.layernorm import (
    LAYERNORM_OUT_FRAC_BITS,
    layernorm_fixed,
    prepare_layernorm,
    prepare_rsqrt_lut,
    rsqrt_fixed,
)
from model.numpy.weights import load_initializers


EMBEDDING_SUM = "/embeddings/Add_1_output_0"
EMBEDDING_OUT = "/embeddings/LayerNorm/LayerNormalization_output_0"


class NumpyEmbeddingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Lấy thêm tensor trung gian của khối embedding làm chuẩn so sánh.
        model = onnx.load(str(config.MODEL_PATH))
        for name in (EMBEDDING_SUM, EMBEDDING_OUT):
            model.graph.output.append(onnx.helper.make_value_info(name, onnx.TypeProto()))
        cls.session = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )

        tensors = load_initializers(config.MODEL_PATH)
        cls.tables, cls.sum_bits = prepare_embeddings(tensors)
        cls.gamma, cls.beta = prepare_layernorm(tensors, "embeddings.LayerNorm")
        cls.lut = prepare_rsqrt_lut()

        rng = np.random.default_rng(0)
        cls.cases = [
            [0, 87, 262, 9384, 2, 1],
            [0, 14343, 4724, 1839, 15091, 7742, 6, 51006, 14849, 5, 2],
            [0, 62, 54063, 25614, 14305, 2558, 13407, 9828, 216348, 1641, 3061, 3941, 32, 2],
            rng.integers(0, 250037, 128).tolist(),
        ]

    def run_cases(self):
        for input_ids in self.cases:
            for token_type in (0, 1):
                input_ids_np = np.array([input_ids], dtype=np.int64)
                token_type_ids = np.full_like(input_ids_np, token_type)
                expected_sum, expected_out = self.session.run(
                    [EMBEDDING_SUM, EMBEDDING_OUT],
                    {
                        "input_ids": input_ids_np,
                        "attention_mask": np.ones_like(input_ids_np),
                        "token_type_ids": token_type_ids,
                    },
                )
                x = embeddings_fixed(input_ids_np, token_type_ids, self.tables, self.sum_bits)
                hidden = layernorm_fixed(x, self.gamma, self.beta, self.lut)

                yield (len(input_ids), token_type), x, hidden, expected_sum, expected_out

    def test_embedding_sum_matches_onnx(self):
        for case, x, _, expected_sum, _ in self.run_cases():
            with self.subTest(case=case):
                self.assertTrue(np.issubdtype(x.dtype, np.integer))
                actual = x / (1 << EMBED_FRAC_BITS)
                self.assertLess(np.abs(actual - expected_sum).max(), 1e-5)

    def test_embedding_layernorm_matches_onnx(self):
        for case, _, hidden, _, expected_out in self.run_cases():
            with self.subTest(case=case):
                self.assertEqual(hidden.dtype, np.int32)
                actual = hidden / (1 << LAYERNORM_OUT_FRAC_BITS)
                cosine = (actual * expected_out).sum(-1) / (
                    np.linalg.norm(actual, axis=-1) * np.linalg.norm(expected_out, axis=-1)
                )
                self.assertLess(np.abs(actual - expected_out).max(), 5e-4)
                self.assertLess(np.abs(actual - expected_out).mean(), 1e-4)
                self.assertGreater(cosine.min(), 0.99999)

    def test_rsqrt_relative_error(self):
        values = np.concatenate([
            np.arange(1, 5000),
            np.random.default_rng(1).integers(1, 1 << 47, 100000),
        ])
        mantissa, shift = rsqrt_fixed(values, self.lut)
        relative = mantissa * np.exp2(-shift.astype(np.float64)) * np.sqrt(values) - 1
        self.assertLess(np.abs(relative).max(), 1e-4)

    def test_layernorm_constant_vector_returns_beta(self):
        x = np.full((1, 2, self.gamma.shape[0]), 12345, dtype=np.int32)
        np.testing.assert_array_equal(
            layernorm_fixed(x, self.gamma, self.beta, self.lut)[0, 0],
            self.beta,
        )


if __name__ == "__main__":
    unittest.main()
