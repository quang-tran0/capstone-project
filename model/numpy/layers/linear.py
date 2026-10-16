import numpy as np

from ..utils.quantize import symmetric_quantize, symmetric_dequantize


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

    def __call__(self, x_quantized, x_scale=None):
        """
        x:      (batch_size, seq_len, in_features), int8
        output: (batch_size, seq_len, out_features), int8
        """
        self.bias_quantized, self.bias_scale = symmetric_quantize(self.bias, x_scale * self.weight_scale)

        y_hat = np.matmul(x_quantized, self.weight_quantized) + self.bias_quantized
        y = symmetric_dequantize(y_hat, x_scale * self.weight_scale)
        y_quantized, y_scale = symmetric_quantize(y)

        # TRACE
        if self.trace and self.layers_output is not None:
            self.layers_output.update({
                f"encoder.layer.{self.name}.output": y,
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
