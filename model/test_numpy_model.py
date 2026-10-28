import math
import unittest

import numpy as np

from model.numpy.config import MODEL_PATH
from model.numpy.layers.int_layer_norm import IntLayerNorm
from model.numpy.layers.linear import Linear
from model.numpy.layers.mean_pooling import MeanPooling
from model.numpy.utils.fixed import FRAC_BITS, to_float
from model.numpy.utils.gelu import gelu_fixed, prepare_phi_lut
from model.numpy.utils.int_quantize import dequantize, quantize_fixed, quantize_weight
from model.numpy.utils.reciprocal import prepare_reciprocal_lut, reciprocal_fixed
from model.numpy.utils.rsqrt import prepare_rsqrt_lut, rsqrt_fixed
from model.numpy.utils.similarity import SCORE_FRAC_BITS, cosine_fixed
from model.numpy.utils.softmax import (
    EXP2_FRAC_BITS,
    PROBABILITY_MAX,
    prepare_exp2_lut,
    softmax_fixed,
)


PREFIX = "encoder.layer.0.output.LayerNorm"
HIDDEN = 384


def fixed(x):
    return np.rint(np.asarray(x) * (1 << FRAC_BITS)).astype(np.int64)


def float_layer_norm(x, weight, bias, eps=1e-12):
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered / np.sqrt((centered ** 2).mean(axis=-1, keepdims=True) + eps) * weight + bias


class IntLayerNormTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ranges match the 25 LayerNorms of MiniLM: |gamma| <= 3.43, |beta| <= 1.82.
        rng = np.random.default_rng(0)
        cls.weight = rng.uniform(0.4, 3.5, HIDDEN)
        cls.bias = rng.uniform(-1.8, 1.8, HIDDEN)
        cls.layer_norm = IntLayerNorm(
            {f"{PREFIX}.weight": cls.weight, f"{PREFIX}.bias": cls.bias}, prefix=PREFIX,
        )
        # Residual-like inputs with a few outlier channels.
        cls.x = rng.normal(0, 1, (2, 32, HIDDEN))
        cls.x[..., rng.choice(HIDDEN, 3, replace=False)] *= 30

    def test_matches_float(self):
        output = self.layer_norm(fixed(self.x))
        self.assertEqual(output.dtype, np.int32)
        actual = to_float(output)
        expected = float_layer_norm(self.x, self.weight, self.bias)
        self.assertLess(np.abs(actual - expected).max(), 2e-3)
        self.assertLess(np.abs(actual - expected).mean(), 2e-4)

    def test_input_scale_cancels(self):
        np.testing.assert_array_equal(
            self.layer_norm(fixed(self.x)),
            self.layer_norm(fixed(self.x) << 4),
        )

    def test_constant_vector_returns_beta(self):
        codes = np.full((1, 2, HIDDEN), 12345, dtype=np.int64)
        np.testing.assert_array_equal(self.layer_norm(codes)[0, 0], self.layer_norm.beta)


class ArithmeticTest(unittest.TestCase):
    def test_rsqrt_relative_error(self):
        values = np.concatenate([
            np.arange(1, 5000),
            np.random.default_rng(1).integers(1, 1 << 47, 100000),
        ])
        mantissa, shift = rsqrt_fixed(values, prepare_rsqrt_lut())
        relative = mantissa * np.exp2(-shift.astype(np.float64)) * np.sqrt(values) - 1
        self.assertLess(np.abs(relative).max(), 1e-4)

    def test_reciprocal_relative_error(self):
        values = np.concatenate([
            np.arange(1, 5000),
            np.random.default_rng(2).integers(1, 1 << 40, 100000),
        ])
        mantissa, shift = reciprocal_fixed(values, prepare_reciprocal_lut())
        relative = mantissa * np.exp2(-shift.astype(np.float64)) * values - 1
        self.assertLess(np.abs(relative).max(), 1e-4)

    def test_quantize_fixed_round_trip(self):
        x = np.random.default_rng(3).normal(0, 2, (4, 16, HIDDEN))
        codes, multiplier, degree = quantize_fixed(fixed(x), axis=-1)
        self.assertEqual(codes.dtype, np.int8)
        self.assertTrue(np.all((multiplier >= 64) & (multiplier <= 127)))
        step = multiplier * np.exp2(-degree.astype(np.float64))
        error = np.abs(dequantize(codes, multiplier, degree) - x)
        self.assertTrue(np.all(error <= 0.51 * step + 2.0**-FRAC_BITS))

    def test_quantize_weight_round_trip(self):
        weight = np.random.default_rng(4).normal(0, 0.05, (HIDDEN, 64))
        codes, multiplier, degree = quantize_weight(weight, axis=0)
        self.assertEqual(multiplier.shape, (1, 64))
        step = multiplier * np.exp2(-degree.astype(np.float64))
        self.assertTrue(np.all(np.abs(dequantize(codes, multiplier, degree) - weight) <= 0.51 * step))

    def test_softmax_matches_float(self):
        rng = np.random.default_rng(5)
        z = rng.normal(0, 3, (2, 12, 20, 20))
        mask = np.ones((2, 1, 1, 20), dtype=bool)
        mask[1, ..., 15:] = False
        codes, total = softmax_fixed(fixed(z), mask, prepare_exp2_lut())
        actual = codes.astype(np.int64) * (1 << EXP2_FRAC_BITS) / (PROBABILITY_MAX * total)
        exponent = np.where(mask, np.exp(z - np.where(mask, z, -np.inf).max(-1, keepdims=True)), 0)
        expected = exponent / exponent.sum(-1, keepdims=True)
        self.assertLess(np.abs(actual - expected).max(), 5e-3)
        self.assertTrue(np.all(codes[1, ..., 15:] == 0))

    def test_gelu_matches_float(self):
        x = np.linspace(-12, 12, 20001)
        expected = x * 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
        actual = to_float(gelu_fixed(fixed(x), prepare_phi_lut()))
        self.assertLess(np.abs(actual - expected).max(), 1e-2)

    def test_linear_matches_float(self):
        rng = np.random.default_rng(6)
        weight = rng.normal(0, 0.05, (HIDDEN, 96))
        bias = rng.normal(0, 0.1, 96)
        x = rng.normal(0, 1, (2, 10, HIDDEN))
        actual = to_float(Linear(weight, bias, "test")(fixed(x)))
        expected = x @ weight + bias
        relative = np.abs(actual - expected).max() / np.abs(expected).max()
        self.assertLess(relative, 2e-2)

    def test_mean_pooling_matches_float(self):
        x = np.random.default_rng(7).normal(0, 1, (2, 9, HIDDEN))
        mask = np.ones((2, 9), dtype=np.int64)
        mask[1, 6:] = 0
        actual = to_float(MeanPooling()(fixed(x), mask))
        expected = (x * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
        self.assertLess(np.abs(actual - expected).max(), 1e-4)

    def test_cosine_matches_float(self):
        rng = np.random.default_rng(8)
        a = rng.normal(0, 0.3, (5, HIDDEN))
        b = rng.normal(0, 0.3, (5, HIDDEN)) + a
        actual = cosine_fixed(fixed(a), fixed(b)) / (1 << SCORE_FRAC_BITS)
        expected = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
        self.assertLess(np.abs(actual - expected).max(), 1e-3)


@unittest.skipUnless(MODEL_PATH.exists(), "needs model/weights/model.onnx")
class EndToEndTest(unittest.TestCase):
    def test_matches_pytorch(self):
        import torch
        from transformers import AutoTokenizer

        from model.numpy.config import MODEL_ID
        from model.numpy.model import SentenceEmbeddingModel as NumpyModel
        from model.pytorch.main import SentenceEmbeddingModel, load_model_from_onnx

        texts = [
            "Tôi đang làm đồ án FPGA.",
            "Đồ án dùng bo mạch KV260 để tăng tốc mô hình nhúng câu.",
            "Hôm nay trời mưa nên tôi ở nhà đọc sách.",
            "Systolic array thực hiện phép nhân ma trận INT8 trên FPGA.",
        ]
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        reference_model = SentenceEmbeddingModel(load_model_from_onnx(MODEL_PATH)).eval()
        numpy_model = NumpyModel(MODEL_PATH)
        reference, actual = [], []
        for text in texts:
            inputs = tokenizer(text, return_tensors="np")
            inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
            with torch.inference_mode():
                reference.append(reference_model(
                    **{name: torch.from_numpy(value) for name, value in inputs.items()}
                ).numpy()[0])
            actual.append(numpy_model(**inputs)[0])
        reference, actual = np.stack(reference), np.stack(actual)

        cosine = (reference * to_float(actual)).sum(-1) / (
            np.linalg.norm(reference, axis=-1) * np.linalg.norm(to_float(actual), axis=-1)
        )
        self.assertGreater(cosine.min(), 0.995)

        reference_scores = reference[1:] @ reference[0] / (
            np.linalg.norm(reference[1:], axis=-1) * np.linalg.norm(reference[0])
        )
        scores = cosine_fixed(actual[:1], actual[1:]) / (1 << SCORE_FRAC_BITS)
        self.assertLess(np.abs(scores - reference_scores).max(), 0.02)
        np.testing.assert_array_equal(np.argsort(-scores), np.argsort(-reference_scores))


if __name__ == "__main__":
    unittest.main()
