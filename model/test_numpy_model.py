import unittest

import numpy as np

from model.numpy.layers.int_layer_norm import OUTPUT_FRAC_BITS, IntLayerNorm
from model.numpy.utils.quantize import symmetric_dequantize, symmetric_quantize
from model.numpy.utils.rsqrt import prepare_rsqrt_lut, rsqrt_fixed
from model.numpy.utils.scale_int12 import ScaleInt12


PREFIX = "encoder.layer.0.output.LayerNorm"
HIDDEN = 384


def float_layer_norm(codes, scale, weight, bias, eps=1e-12):
    """Float64 reference: the IntLayerNorm formula before the integer rsqrt."""
    centered = codes * codes.shape[-1] - codes.sum(axis=-1, keepdims=True)
    variance = np.mean(centered.astype(np.float64) ** 2, axis=-1, keepdims=True)
    denominator = np.sqrt(variance + eps * (codes.shape[-1] / float(scale)) ** 2)
    return centered / denominator * weight + bias


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

        # Residual-like wide accumulators with a few outlier channels.
        cls.cases = []
        for degree in (14, 20, 26):
            x = rng.normal(0, 1, (2, 32, HIDDEN))
            x[..., rng.choice(HIDDEN, 3, replace=False)] *= 30
            codes = np.rint(x * 0.01 * 2**degree).astype(np.int64)
            cls.cases.append((codes, ScaleInt12.from_parts(1, degree, normalize=False)))

    def test_normalize_matches_float(self):
        for codes, scale in self.cases:
            with self.subTest(degree=scale.degree):
                output = self.layer_norm.normalize(codes)
                self.assertEqual(output.dtype, np.int32)
                actual = output / (1 << OUTPUT_FRAC_BITS)
                expected = float_layer_norm(codes, scale, self.weight, self.bias)
                cosine = (actual * expected).sum(-1) / (
                    np.linalg.norm(actual, axis=-1) * np.linalg.norm(expected, axis=-1)
                )
                self.assertLess(np.abs(actual - expected).max(), 2e-3)
                self.assertLess(np.abs(actual - expected).mean(), 2e-4)
                self.assertGreater(cosine.min(), 0.99999)

    def test_int8_output_within_one_lsb(self):
        for codes, scale in self.cases:
            with self.subTest(degree=scale.degree):
                quantized, output_scale = self.layer_norm((codes, scale))
                self.assertEqual(quantized.dtype, np.int8)
                self.assertIsInstance(output_scale, ScaleInt12)
                expected_q, expected_scale = symmetric_quantize(
                    float_layer_norm(codes, scale, self.weight, self.bias)
                )
                difference = np.abs(
                    symmetric_dequantize(quantized, output_scale)
                    - symmetric_dequantize(expected_q, expected_scale)
                )
                self.assertLessEqual(difference.max(), float(expected_scale) * 1.01)

    def test_rsqrt_relative_error(self):
        values = np.concatenate([
            np.arange(1, 5000),
            np.random.default_rng(1).integers(1, 1 << 47, 100000),
        ])
        mantissa, shift = rsqrt_fixed(values, prepare_rsqrt_lut())
        relative = mantissa * np.exp2(-shift.astype(np.float64)) * np.sqrt(values) - 1
        self.assertLess(np.abs(relative).max(), 1e-4)

    def test_constant_vector_returns_beta(self):
        codes = np.full((1, 2, HIDDEN), 12345, dtype=np.int64)
        np.testing.assert_array_equal(
            self.layer_norm.normalize(codes)[0, 0],
            self.layer_norm.beta,
        )


if __name__ == "__main__":
    unittest.main()
