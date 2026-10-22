import numpy as np

from ..utils.quantize import symmetric_quantize, symmetric_dequantize, requantize
from ..utils.scale_int12 import ScaleInt12


class Linear:
    def __init__(self, weight, bias, name, trace=False, layers_output=None):
        # Weight lấy trực tiếp từ ONNX: (in_features, out_features).
        self.weight = np.asarray(weight, dtype=np.float32)
        self.weight_quantized, self.weight_scale = symmetric_quantize(self.weight)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.bias_quantized, self.bias_scale = None, None

        self.name = name
        self.trace = trace
        self.layers_output = layers_output

    def __call__(self, x_quantized, x_scale):
        """
        x:      (batch_size, seq_len, in_features), codes in [-127, 127]
        output: (batch_size, seq_len, out_features), np.int8 + ScaleInt12
        """
        x_quantized = np.asarray(x_quantized)
        if not np.issubdtype(x_quantized.dtype, np.integer):
            raise TypeError("Linear input must contain integer codes")
        if np.any(x_quantized < -127) or np.any(x_quantized > 127):
            raise ValueError("Linear input codes must be in [-127, 127]")
        if not isinstance(x_scale, ScaleInt12) or float(x_scale) <= 0:
            raise ValueError("Linear input scale must be a positive ScaleInt12")
        accumulator_scale = x_scale.multiply_wide(self.weight_scale)
        self.bias_quantized, self.bias_scale = symmetric_quantize(
            self.bias, accumulator_scale, type=np.int32
        )
        y_hat = (
            np.matmul(x_quantized.astype(np.int64), self.weight_quantized.astype(np.int64))
            + self.bias_quantized.astype(np.int64)
        )
        y_quantized, y_scale = requantize(y_hat, accumulator_scale)

        # TRACE
        if self.trace and self.layers_output is not None:
            self.layers_output.update({
                f"encoder.layer.{self.name}.accumulator": y_hat,
                f"encoder.layer.{self.name}.accumulator_scale": accumulator_scale,
                f"encoder.layer.{self.name}.output": symmetric_dequantize(y_quantized, y_scale),
                f"encoder.layer.{self.name}.quantized": y_quantized,
                f"encoder.layer.{self.name}.scale": y_scale
            })

        return y_quantized, y_scale

    def display_trace(self):
        if self.trace and self.layers_output is not None:
            print(f"\n===== Linear Layer Info ({self.name}) =====")
            print("- Output max:", np.max(self.layers_output[f"encoder.layer.{self.name}.output"]))
            print("- Output min:", np.min(self.layers_output[f"encoder.layer.{self.name}.output"]))
            print("- Output mean:", np.mean(self.layers_output[f"encoder.layer.{self.name}.output"]))
            print("- Scale:", self.layers_output[f"encoder.layer.{self.name}.scale"])
