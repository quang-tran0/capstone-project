import unittest

import numpy as np

from .ops import dynamic_quantize, gelu, layer_norm, quantized_linear, softmax


class NumericalOperationsTest(unittest.TestCase):
    def test_dynamic_quantize_handles_constant_zero_input(self):
        quantized, scale, zero_point = dynamic_quantize(
            np.zeros((2, 3), dtype=np.float32)
        )

        np.testing.assert_array_equal(quantized, np.zeros((2, 3), dtype=np.uint8))
        self.assertEqual(quantized.dtype, np.uint8)
        self.assertEqual(scale, np.float32(1.0))
        self.assertEqual(zero_point, np.uint8(0))

    def test_quantized_linear_uses_integer_accumulation_and_rescaling(self):
        output = quantized_linear(
            np.array([[0.0, 1.0]], dtype=np.float32),
            np.array([[2, -1], [1, 3]], dtype=np.int8),
            np.float32(0.5),
            np.int8(0),
            np.array([1.0, -2.0], dtype=np.float32),
        )

        np.testing.assert_allclose(
            output,
            np.array([[1.5, -0.5]], dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )
        self.assertEqual(output.dtype, np.float32)

    def test_layer_norm_normalizes_last_dimension(self):
        output = layer_norm(
            np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
            np.ones(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.float32(1e-12),
        )

        self.assertAlmostEqual(float(output.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(output.var()), 1.0, places=6)
        self.assertEqual(output.dtype, np.float32)

    def test_softmax_is_stable_for_large_negative_values(self):
        output = softmax(
            np.array([[-10000.0, -10001.0], [-3.0, -3.0]], dtype=np.float32)
        )

        np.testing.assert_allclose(output.sum(axis=-1), np.ones(2), atol=1e-6)
        self.assertTrue(np.isfinite(output).all())
        self.assertEqual(output.dtype, np.float32)

    def test_gelu_uses_exact_error_function(self):
        output = gelu(np.array([0.0, 1.0], dtype=np.float32))

        np.testing.assert_allclose(
            output,
            np.array([0.0, 0.8413447], dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )
        self.assertEqual(output.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
